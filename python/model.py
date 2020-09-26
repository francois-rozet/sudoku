#!/usr/bin/env python


###########
# Imports #
###########

import csv
import os
import torch
import torch.nn as nn

from PIL import Image
from torch.utils import data
from torchvision import datasets

from generator import base64ToPIL


#############
# Functions #
#############

def DoubleConvolution(in_channels, out_channels, kernel_size=3, padding=1):
	'''Generic double convolution layer'''
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True),
		nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)


def FullConnection(in_channels, out_channels):
	'''Generic full connection layer'''
	return nn.Sequential(
		nn.Linear(in_channels, out_channels),
		nn.ReLU(inplace=True)
	)


###########
# Classes #
###########

class RozNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.C1 = DoubleConvolution(1, 16, 5, 2)
		self.S1 = nn.MaxPool2d(2)

		self.C2 = DoubleConvolution(16, 32, 5, 2)
		self.S2 = nn.MaxPool2d(2)

		self.C3 = DoubleConvolution(32, 64, 5, 2)
		self.S3 = nn.MaxPool2d(2)

		self.F1 = FullConnection(576, 288)
		self.F2 = FullConnection(288, 144)

		self.last = nn.Linear(144, 11)
		self.soft = nn.Softmax(dim=1)

	def head(self, x):
		x = self.last(x)

		return x if self.training else self.soft(x)

	def forward(self, x):
		# Features extraction
		x = self.C1(x)
		x = self.S1(x)

		x = self.C2(x)
		x = self.S2(x)

		x = self.C3(x)
		x = self.S3(x)

		# Classification
		x = torch.flatten(x, 1)

		x = self.F1(x)
		x = self.F2(x)

		return self.head(x)


class GaussianNoise(nn.Module):
	'''Pixelwise Gaussian noise transform.'''

	def __init__(self, mean=0., std=1.):
		super().__init__()

		self.std = std
		self.mean = mean

	def forward(self, x):
		return torch.clamp(
			x + torch.randn(x.size()) * self.std + self.mean,
			min=0.,
			max=1.
		)


class AQMNIST(data.Dataset):
	"""Augmented QMNIST.

	The augmentation consists in adding empty images, i.e. without digit,
	as well as printed digits (not handwritten) with various fonts.

	References
	----------
	Cold Case: The Lost MNIST Digits
	(Yadav C. and Bottou L., 2019)
	https://arxiv.org/abs/1905.10498
	"""

	black = Image.new('L', (28, 28))

	def __init__(self, transform=lambda x: x, what='train'):
		super().__init__()

		self.transform = transform
		self.qmnist = datasets.QMNIST(root='resources/qmnist/', what=what, download=True)

		self.printed = []

		if os.path.exists('resources/csv/printed_digits.csv'):
			with open('resources/csv/printed_digits.csv', 'r') as f:
				reader = csv.reader(f, delimiter=',')

				for row in reader:
					self.printed.append((
						base64ToPIL(row[2]).convert('L'),
						int(row[1])
					))

		self.printed = [
			x
			for i, x in enumerate(self.printed)
			if (i % 5 == 0) == (what == 'test')
		]

	def __len__(self):
		return len(self.qmnist) + len(self.printed) + len(self.qmnist) // 10

	def __getitem__(self, i):
		if i < len(self.qmnist):
			inpt, targt = self.qmnist[i]
		elif i < len(self.qmnist) + len(self.printed):
			inpt, targt = self.printed[i - len(self.qmnist)]
		else:
			inpt, targt = self.black, 10

		return self.transform(inpt), targt


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import argparse
	import numpy as np
	import time

	from torch.optim import Adam, lr_scheduler
	from torchvision import transforms

	# Arguments
	parser = argparse.ArgumentParser(description='Train model')
	parser.add_argument('-o', '--output', default='products/weights/roznet.pth', help='output weights file')
	parser.add_argument('-bsize', type=int, default=64, help='batch size')
	parser.add_argument('-epochs', type=int, default=15, help='number of epochs')
	parser.add_argument('-step', type=int, default=5, help='step size')
	parser.add_argument('-gamma', type=float, default=1e-1, help='gamma')
	parser.add_argument('-lrate', type=float, default=1e-2, help='learning rate')
	parser.add_argument('-wdecay', type=float, default=0., help='weight decay')
	args = parser.parse_args()

	# Model
	model = RozNet()

	criterion = nn.CrossEntropyLoss()
	optimizer = Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

	if torch.cuda.is_available():
		model = model.cuda()
		criterion = criterion.cuda()

	# Training set
	norm = transforms.ToTensor()

	transform = transforms.Compose([
		transforms.ColorJitter(brightness=0.2, contrast=0.2),
		transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
		norm,
		GaussianNoise(mean=0., std=0.15)
	])

	trainset = AQMNIST(transform=transform, what='train')
	trainloader = data.DataLoader(trainset, batch_size=args.bsize, shuffle=True)

	validset = AQMNIST(transform=norm, what='test')
	validloader = data.DataLoader(validset, batch_size=args.bsize, shuffle=True)

	# Training
	for epoch in range(args.epochs):
		print('-' * 10)
		print('Epoch {}, lr = {}'.format(epoch, scheduler.get_last_lr()[0]))

		start = time.time()

		model.train()

		losses = []

		for inputs, targets in trainloader:
			if torch.cuda.is_available():
				inputs = inputs.cuda()
				targets = targets.cuda()

			outputs = model(inputs)
			loss = criterion(outputs, targets)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses.append(loss.item())

		losses = np.array(losses)

		print('Epoch average training loss = {}'.format(losses.mean()))

		model.eval()

		losses = []

		for inputs, targets in validloader:
			if torch.cuda.is_available():
				inputs = inputs.cuda()
				targets = targets.cuda()

			outputs = model(inputs)
			loss = torch.mean((torch.argmax(outputs, dim=1) == targets).double())

			losses.append(loss.item())

		losses = np.array(losses)

		print('Epoch average validation accuracy = {}'.format(losses.mean()))

		elapsed = time.time() - start

		print('{:.0f}m{:.0f}s elapsed'.format(elapsed // 60, elapsed % 60))

		scheduler.step()

	# Save
	os.makedirs(os.path.dirname(args.output), exist_ok=True)
	torch.save(model.state_dict(), args.output)
