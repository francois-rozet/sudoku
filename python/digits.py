#!/usr/bin/env python


###########
# Imports #
###########

import cv2
import numpy as np
import scipy.interpolate as spi
import torch

from torchvision import transforms


#############
# Functions #
#############

def load(imagepath, maxsize=500):
	"""Load an image given its path.

	Parameters
	----------
	name : str
		image path
	maxsize : int
		maximum size of the shortest dimension

	Returns
	-------
	numpy.uint8
		image
	"""

	img = cv2.imread(imagepath)

	while min(img.shape[:2]) > maxsize:
		img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

	return img


def preprocess(img, dsize=11):
	"""Adjust image brightness.

	Parameters
	----------
	img : numpy.uint8
		image
	dsize : int
		morphological closing disk kernel size

	Returns
	-------
	numpy.uint8
		preprocessed gray scale image
	"""

	# Convert to gray scale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Normalize brightness
	close = cv2.morphologyEx(
		gray,
		cv2.MORPH_CLOSE,
		cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dsize, dsize))
	)

	div = (np.uint16(gray) * 256) / np.uint16(close)
	img = np.uint8(cv2.normalize(div, None, 0, 255, cv2.NORM_MINMAX))

	return img


def detect(img):
	"""Detect and isolate Sudoku grid in image.

	Parameters
	----------
	img : numpy.uint8
		image

	Returns
	-------
	numpy.uint8
		isolated grid image
	"""

	# Detect grid countour
	_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	grid = None
	area = np.array(img.shape).prod() / 4 # at least a 4th of the image

	for cnt in cnts:
		temp = cv2.contourArea(cnt)
		if temp > area:
			grid = cnt
			area = temp

	if grid is None:
		return None

	# Isolate grid
	mask = np.zeros(img.shape, dtype=np.uint8)
	cv2.drawContours(mask, [grid], 0, 255, -1)
	img = img.copy()
	img[mask == 0] = 255

	# Rotate
	center, _, angle = cv2.minAreaRect(grid)

	if angle < -45.:
		angle += 90.

	M = cv2.getRotationMatrix2D(center, angle, 1)
	img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=255)

	return img


def lines(img, ksize=3, w=2, h=5):
	"""Detect vertical edges.

	Parameters
	----------
	img : numpy.uint8
		image
	ksize : int
		Sobel derivative kernel size
	w : int
		morphological dilation rectangle kernel width
	h : int
		morphological dilation rectangle kernel height

	Yields
	------
	numpy.uint8
		vertical edge mask
	"""

	# Vertical edge detection
	img = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=ksize)
	_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (w, h)), iterations=1)

	# Connected component analysis
	criteria = min(img.shape)

	cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	lines = []

	for cnt in cnts:
		if cv2.arcLength(cnt, True) > criteria:
			(x, y), _, _ = cv2.minAreaRect(cnt)
			lines.append((x, cnt))

	# Sorting from left to right
	lines.sort(key=lambda x: x[0])

	for _, cnt in lines:
		mask = np.zeros(img.shape, dtype=np.uint8)
		cv2.drawContours(mask, [cnt], 0, 255, -1)
		mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h)), iterations=2)
		yield mask


def straighten(img, n=9, px=50):
	"""Straighten grid and warp image.

	Parameters
	----------
	img : numpy.uint8
		image
	n : int
		number of cells per row
	px : int
		number of pixels per row

	Returns
	-------
	numpy.uint8
		straightened grid image
	"""

	# Intersections
	intersections = []
	anchors = []

	hlines = list(lines(img.T))
	vlines = list(lines(img))

	for i, hl in enumerate(hlines):
		for j, vl in enumerate(vlines):
			mask = cv2.bitwise_and(hl.T, vl)

			cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			if not cnts:
				continue

			x, y = np.min(cnts[0][:, 0, :], axis=0).astype(int)
			intersections.append([x, y])
			anchors.append([i * px, j * px])

	intersections = np.array(intersections)
	anchors = np.array(anchors)

	# Interpolation
	mapping = spi.griddata(
		anchors,
		intersections,
		np.array(np.meshgrid(np.arange(0, n * px + 1), np.arange(0, n * px + 1))).T.reshape(-1, 2),
		method='cubic'
	).astype('float32')

	img = cv2.remap(
		img,
		mapping[:, 0].reshape(n * px + 1, n * px + 1),
		mapping[:, 1].reshape(n * px + 1, n * px + 1),
		cv2.INTER_CUBIC
	)

	return img


def split(img, nrows=9, ncols=9, padding=(3, 3), reshape=None):
	"""Split image.

	Parameters
	----------
	img : numpy.uint8
		image
	nrows : int
	rcols : int
	padding : (int, int)

	Yields
	------
	numpy.uint8
		cell image
	"""

	# Split
	width = img.shape[1] // nrows
	height = img.shape[0] // ncols

	for y in range(nrows):
		ymin = y * height + padding[1]
		ymax = (y + 1) * height - padding[1]

		for x in range(ncols):
			xmin = x * width + padding[0]
			xmax = (x + 1) * width - padding[0]

			split = img[ymin:ymax, xmin:xmax]

			if reshape is not None:
				split = cv2.resize(split, reshape)

			yield split


def digits(img, model):
	"""Classify grid digits.

	Parameters
	----------
	img : numpy.uint8
		image
	model : torch.nn.Module
		digit classifier

	Returns
	-------
	numpy.array
		grid digits
	"""

	transform = transforms.ToTensor()

	model.eval()

	with torch.no_grad():
		batch = torch.stack([
			transform(255 - x) for x in split(img, reshape=(28, 28))
		])

		if torch.cuda.is_available():
			batch = batch.cuda()

		output = model(batch)

	dgts = torch.argmax(output, dim=1).cpu().reshape(9, 9).numpy()
	dgts[dgts == 10] = 0

	return dgts


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import sys

	from model import RozNet

	# Model
	model = RozNet()

	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	model = model.to(device)
	model.load_state_dict(torch.load('products/weights/roznet.pth', map_location=device))

	# Process
	verbose = False

	for imgname in sys.argv[1:]:
		if imgname == '-v':
			verbose = True
			continue

		img = load(imgname)

		if verbose:
			cv2.imshow('0 - Original', img)
			cv2.waitKey()

		img = preprocess(img)

		if verbose:
			cv2.imshow('1 - Preprocessed', img)
			cv2.waitKey()

		img = detect(img)

		if verbose:
			cv2.imshow('2 - Isolated', img)
			cv2.waitKey()

		img = straighten(img)

		if verbose:
			cv2.imshow('3 - Straightened', img)
			cv2.waitKey()

		# Digits
		dgts = digits(img, model)
		np.savetxt(sys.stdout, dgts, fmt='%d', delimiter=' ')

		if imgname != sys.argv[-1]:
			print()
