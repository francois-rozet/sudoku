#!/usr/bin/env python


###########
# Imports #
###########

import numpy as np


###########
# Classes #
###########

class Cell:
	def __init__(self, index, zone, value, domain):
		self.index = index
		self.zone = zone
		self.value = value

		self.domain = domain

		self.neighbors = []
		self.options = set()

	def __len__(self):
		return len(self.options)

	def update(self):
		self.options = self.domain.copy()

		for x in self.neighbors:
			if x.value > 0:
				self.options.discard(x.value)


class Solver:
	def __init__(self, grid, shuffle=False):
		self.grid = grid

		# Parameters
		n = int(self.grid.shape[0] ** (1. / self.grid.ndim))
		domain = set(range(1, 1 + self.grid.shape[0]))

		# Cells
		self.cells, self.free, self.guesses = [], [], []

		for index, value in np.ndenumerate(self.grid):
			self.cells.append(Cell(
				index,
				tuple(x // n for x in index),
				value,
				domain
			))

		for x in self.cells:
			for y in self.cells:
				if x.index == y.index:
					break
				elif x.zone == y.zone or sum(a != b for a, b in zip(x.index, y.index)) == 1:
					x.neighbors.append(y)
					y.neighbors.append(x)

		for x in self.cells:
			if x.value == 0:
				self.free.append(x)
				x.update()

		if shuffle:
			np.random.shuffle(self.free)

	def __iter__(self):
		if self.free:
			self.free.sort(key=len, reverse=True)
			self.guesses.append(self.free.pop())

		while self.guesses:
			cell = self.guesses.pop()

			# Update cell value
			cell.value = cell.options.pop() if cell.options else 0

			for x in self.free:
				if x in cell.neighbors:
					x.update()

			# Select next cell
			if cell.value > 0:
				self.guesses.append(cell)

				if self.free:
					self.free.sort(key=len, reverse=True)
					self.guesses.append(self.free.pop())
				else:
					yield self.fill()
			else:
				self.free.append(cell)
				cell.update()

	def fill(self):
		grid = self.grid.copy()
		for cell in self.guesses:
			grid[cell.index] = cell.value

		return grid


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import argparse
	import sys

	# Parser
	parser = argparse.ArgumentParser(description='Sudoku Solver')
	parser.add_argument('-f', '--file')
	parser.add_argument('-s', '--sep', default=' ')
	parser.add_argument('-o', '--output', default=None)
	parser.add_argument('-n', '--number', type=int, default=1)
	args = parser.parse_args()

	if args.output is not None:
		sys.stdout = open(args.output, 'w')

	# Load
	grid = np.loadtxt(args.file, dtype=int, delimiter=args.sep)

	# Solutions
	for i, solution in enumerate(Solver(grid)):
		if i == args.number:
			break
		elif i > 0:
			print(end='\n')

		np.savetxt(sys.stdout, solution, fmt='%d', delimiter=args.sep)
