#!/usr/bin/env python


###########
# Imports #
###########

import base64
import cairo
import io
import numpy as np
import random

from PIL import Image


#############
# Functions #
#############

def findAllFonts():
	import matplotlib.font_manager as fm
	from fontTools.ttLib import TTFont

	fonts = set()

	for font in fm.findSystemFonts():
		if font.endswith(('.ttf', '.TTF')):
			for table in TTFont(font)['cmap'].tables:
				if table.cmap.get(ord('0')) == 'zero':
					fonts.add(
						fm.FontProperties(fname=font).get_name()
					)
					break

	return list(fonts)


def base64ToPIL(x):
	return Image.open(io.BytesIO(base64.b64decode(x.encode())))


def base64FromPIL(x):
	f = io.BytesIO()
	x.save(f, format='PNG')
	return base64.b64encode(f.getvalue()).decode()


def buildPrintedDigits(filename, size=28):
	import csv

	with open(filename, 'w', newline='') as f:
		writer = csv.writer(f)

		with cairo.ImageSurface(cairo.Format.RGB24, size, size) as surface:
			ctx = cairo.Context(surface)
			ctx.scale(size, size)

			for font in findAllFonts():
				## Select Font
				ctx.select_font_face(
					font,
					cairo.FONT_SLANT_NORMAL,
					cairo.FONT_WEIGHT_NORMAL
				)
				ctx.set_font_size(0.9)

				for digit in range(10):
					### Fill background
					ctx.rectangle(0, 0, 1, 1)
					ctx.set_source_rgb(0, 0, 0)
					ctx.fill()
					ctx.set_source_rgb(1, 1, 1)

					### Write digit
					_, _, width, height, _, _ = ctx.text_extents(str(digit))
					ctx.move_to(0.5 - width / 2, 0.5 + height / 2)
					ctx.show_text(str(digit))

					### Export
					g = io.BytesIO()
					surface.write_to_png(g)
					img = Image.open(g).convert('L')

					writer.writerow([
						font,
						digit,
						base64FromPIL(img)
					])


def draw(grid, font, size=1024):
	with cairo.ImageSurface(cairo.Format.RGB24, size, size) as surface:
		ctx = cairo.Context(surface)
		ctx.scale(size, size)

		# Parameters
		n = int(np.sqrt(grid.shape[0]))

		start = 0.1
		end = 0.9

		step = (end - start) / n ** 2

		# Background
		ctx.rectangle(0, 0, 1, 1)
		ctx.set_source_rgb(1, 1, 1)
		ctx.fill()

		# Noise
		ctx.save()
		ctx.rotate((random.random() - 1 / 2) * np.pi / 18)
		ctx.rectangle(
			random.random() * 2 * start,
			random.random() * 2 * start,
			1.05 * (end - start),
			1.05 * (end - start)
		)
		ctx.set_source_rgb(0.6 + random.random() * 0.4, 0.6 + random.random() * 0.4, 0.6 + random.random() * 0.4)
		ctx.fill()
		ctx.restore()

		# Grid
		ctx.rectangle(start, start, end - start, end - start)
		ctx.set_source_rgb(1, 1, 1)
		ctx.fill_preserve()
		ctx.set_source_rgb(0, 0, 0)
		ctx.set_line_width(0.01)
		ctx.stroke()

		for i, x in enumerate(np.arange(start, end, step)):
			if i == 0:
				continue

			if i % n == 0:
				ctx.set_line_width(0.01)
			else:
				ctx.set_line_width(0.005)

			ctx.move_to(start, x)
			ctx.line_to(end, x)
			ctx.stroke()

			ctx.move_to(x, start)
			ctx.line_to(x, end)
			ctx.stroke()

		# Numbers
		ctx.select_font_face(
			font,
			cairo.FONT_SLANT_NORMAL,
			cairo.FONT_WEIGHT_NORMAL
		)
		ctx.set_font_size(0.09 * (end - start))

		for (i, j), number in np.ndenumerate(grid):
			if number == 0:
				continue

			_, _, width, height, _, _ = ctx.text_extents(str(number))

			ctx.move_to(start + (j + 1 / 2) * step - width / 2, start + (i + 1 / 2) * step + height / 2)
			ctx.show_text(str(number))

		f = io.BytesIO()
		surface.write_to_png(f)

	return Image.open(f)


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import argparse
	import os

	from solver import Sudoku

	# Parser
	parser = argparse.ArgumentParser(description='Sudoku Solver')
	parser.add_argument('-d', '--destination', default='images')
	parser.add_argument('-o', '--output', default='sudoku')
	parser.add_argument('-n', '--number', type=int, default=100)
	args = parser.parse_args()

	os.makedirs(args.destination, exist_ok=True)
	filename = os.path.join(args.destination, args.output) + '_{:04}'

	# Fonts
	fonts = findAllFonts()

	# Grids
	for i in range(1, args.number + 1):
		for solution in Sudoku(n=3, d=2).solve(shuffle=True):
			break

		solution = Sudoku(grid=solution).unsolve()

		## Draw
		img = draw(solution, random.choice(fonts))

		## Save
		basename = filename.format(i)

		img.save(basename + '.jpg')
		np.savetxt(basename + '.dat', solution, fmt='%d')
