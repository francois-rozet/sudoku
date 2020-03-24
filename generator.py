#!/usr/bin/env python


###########
# Imports #
###########

import cairo
import numpy as np
import random


##############
# Parameters #
##############

FONTS = [
	'Arial',
	'Cambria',
	'Comic Sans MS',
	'Courier New',
	'Ink Free',
	'Lato',
	'Segoe Print'
]


#############
# Functions #
#############

def draw(filename, grid, size=1024):
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
			random.choice(FONTS),
			cairo.FONT_SLANT_NORMAL,
			cairo.FONT_WEIGHT_NORMAL
		)
		ctx.set_font_size(0.09 * (end - start))

		for (i, j), number in np.ndenumerate(grid):
			if number == 0:
				continue

			x, y, width, height, dx, dy = ctx.text_extents(str(number))

			ctx.move_to(start + (j + 1 / 2) * step - width / 2, start + (i + 1 / 2) * step + height / 2)
			ctx.show_text(str(number))

		surface.write_to_png(filename)

	return filename


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import argparse
	import os

	from solver import Solver, Unsolver

	# Parser
	parser = argparse.ArgumentParser(description='Sudoku Solver')
	parser.add_argument('-d', '--destination', default='images')
	parser.add_argument('-o', '--output', default='sudoku')
	parser.add_argument('-n', '--number', type=int, default=100)
	args = parser.parse_args()

	os.makedirs(args.destination, exist_ok=True)
	filename = os.path.join(args.destination, args.output) + '_{:04}'

	# Grids
	grid = np.zeros((9, 9), dtype=int)

	for i in range(1, args.number + 1):
		for solution in Solver(grid, shuffle=True):
			break

		solution = Unsolver(solution)()

		## Save
		basename = filename.format(i)
		np.savetxt(basename + '.dat', solution, fmt='%d')
		draw(basename + '.png', solution)
