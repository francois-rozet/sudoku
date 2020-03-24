#!/usr/bin/env python


###########
# Imports #
###########

import numpy as np


###########
# Classes #
###########

class Cell:
	def __init__(self, index, grid, domain):
		self.index = index
		self.grid = grid
		self.value = self.grid[self.index]

		self.domain = domain

		self.families = []
		self.options = set()

	def __len__(self):
		return len(self.options)

	def empty(self):
		return self.value == 0

	def valid(self):
		return self.empty() or self.value in self.options

	def update(self):
		self.options = self.domain.copy()

		for family in self.families:
			for cell in family:
				if cell != self and not cell.empty():
					self.options.discard(cell.value)

	def renew(self, value):
		self.value = self.grid[self.index] = value

		for family in self.families:
			for cell in family:
				if cell.empty():
					cell.update()


class Sudoku:
	def __init__(self, grid, exceptions={}):
		# Grid
		self.grid = grid.copy()

		ndim = len(self.grid.shape)
		N = self.grid.shape[0]
		n = int(N ** (1. / ndim))
		domain = set(range(1, 1 + N))

		# Cells
		self.cells = []

		for index, value in np.ndenumerate(self.grid):
			exception = exceptions.get(index)
			self.cells.append(Cell(
				index,
				self.grid,
				domain.difference(exception) if exception else domain
			))

		# Families
		for i in range(ndim):
			rows = [set() for _ in range(N)]
			for cell in self.cells:
				j = cell.index[i]
				rows[j].add(cell)
				cell.families.append(rows[j])

		blocks = [set() for _ in range(N)]
		for cell in self.cells:
			j = sum((x // n) * n ** i for i, x in enumerate(cell.index))
			blocks[j].add(cell)
			cell.families.append(blocks[j])

		for cell in self.cells:
			cell.update()

	def valid(self):
		return all(cell.valid() for cell in self.cells)


class Solver:
	def __init__(self, grid, shuffle=True, exceptions={}):
		self.sudoku = Sudoku(grid, exceptions)

		# Free cells
		self.free = list(filter(
			lambda x: x.empty(),
			self.sudoku.cells
		))
		self.guesses = []

		if shuffle:
			np.random.shuffle(self.free)

	def __iter__(self):
		if not self.sudoku.valid():
			raise ValueError('Invalid Grid')

		if self.free:
			self.free.sort(key=len, reverse=True)
			self.guesses.append(self.free.pop())

		while self.guesses:
			cell = self.guesses.pop()

			# Renew cell value
			cell.renew(cell.options.pop() if cell.options else 0)

			# Select next cell
			if not cell.empty():
				self.guesses.append(cell)

				if self.free:
					self.free.sort(key=len, reverse=True)
					self.guesses.append(self.free.pop())
				else:
					yield self.sudoku.grid
			else:
				self.free.append(cell)


class Unsolver:
	def __init__(self, grid, shuffle=True):
		self.sudoku = Sudoku(grid)

		# Filled cells
		self.filled = list(filter(
			lambda x: not x.empty(),
			self.sudoku.cells
		))

		if shuffle:
			np.random.shuffle(self.filled)

	def __call__(self):
		if not self.sudoku.valid():
			raise ValueError('Invalid Grid')

		self.filled.sort(key=len, reverse=True)

		while self.filled:
			cell = self.filled.pop()
			value = cell.value
			cell.renew(0)

			if len(cell.options) > 1:
				n = 0
				for _ in Solver(self.sudoku.grid, exceptions={cell.index: {value}}):
					n += 1
					break

				if n > 0:
					cell.renew(value)

		return self.sudoku.grid


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import argparse
	import sys

	# Parser
	parser = argparse.ArgumentParser(description='Sudoku Solver')
	parser.add_argument('-f', '--file', default=None)
	parser.add_argument('-s', '--sep', default=' ')
	parser.add_argument('-o', '--output', default=None)
	parser.add_argument('-n', '--number', type=int, default=1)
	parser.add_argument('-u', '--unsolved', default=False, action='store_true')
	args = parser.parse_args()

	if args.output is not None:
		sys.stdout = open(args.output, 'w')

	# Load
	if args.file is None:
		grid = np.zeros((9, 9), dtype=int)
	else:
		grid = np.loadtxt(args.file, dtype=int, delimiter=args.sep)

	# Solutions
	if args.unsolved and args.file is not None:
		np.savetxt(
			sys.stdout,
			Unsolver(grid)(),
			fmt='%0{}d'.format(int(np.log10(grid.shape[0])) + 1),
			delimiter=args.sep
		)
	else:
		for i, solution in enumerate(Solver(grid)):
			if i == args.number:
				break
			elif i > 0:
				print(end='\n')

			np.savetxt(
				sys.stdout,
				Unsolver(solution)() if args.unsolved else solution,
				'%0{}d'.format(int(np.log10(grid.shape[0])) + 1),
				delimiter=args.sep
			)
