#!/usr/bin/env python


###########
# Imports #
###########

import numpy as np


###########
# Classes #
###########

class Sudoku:
	def __init__(self, n=3, d=2, grid=None):
		if grid is None:
			self.n = n
			self.d = d
			self.N = n ** d
		else:
			self.N = grid.shape[0]
			self.d = len(grid.shape)
			self.n = int(self.N ** (1. / self.d))

		self.grid = np.zeros((self.N,) * self.d, dtype=int)
		self.counter = np.zeros(self.grid.shape + (self.N + 1,), dtype=int)
		self.counter[..., 0] = self.N

		if grid is not None:
			for key, value in np.ndenumerate(grid):
				self[key] = value

	def __len__(self):
		return self.grid.size

	def __iter__(self):
		return np.ndindex(self.grid.shape)

	def __getitem__(self, key):
		'''Get cell's value.'''
		return self.grid[key]

	def __setitem__(self, key, value):
		'''Set cell's value.'''

		if not self.valid(key, value):
			print(self.grid)
			print(self.counter[key])
			raise ValueError('Invalid value {} for cell {}'.format(value, key))

		# Update rows counters
		for i in range(self.d):
			row = self.counter[key[:i] + (slice(self.N),) + key[i+1:]]

			row[:, self[key]] -= 1
			if value != 0:
				row[:, value] += 1

			## Number of options
			row[:, 0] = np.sum(row[:, :] == 0, axis=1)

		# Update block counters
		temp = np.array(key)
		temp -= temp % self.n
		block = self.counter[tuple([slice(i, j) for i, j in zip(temp, temp + self.n)])]

		block[..., self[key]] -= 1
		if value != 0:
			block[..., value] += 1

		## Number of options
		block[..., 0] = np.sum(block == 0, axis=self.d)

		# Update cell
		self.counter[key][self[key]] = 0
		if value != 0:
			self.counter[key][value] = 0

		## Number of options
		self.counter[key][0] = np.sum(self.counter[key] == 0)

		self.grid[key] = value

	def empty(self, key):
		'''State if cell is empty.'''
		return self[key] == 0

	def options(self, key):
		'''Get cell's value options.'''
		return (np.argwhere(self.counter[key][1:] == 0).ravel() + 1).tolist()

	def valid(self, key, value):
		'''State if value is a valid cell's value option.'''
		return value == 0 or self.counter[key][value] == 0

	def solve(self, shuffle=True):
		'''Iterate over all solutions.'''
		guesses = []
		backprop = False

		while True:
			if not backprop:
				# If current grid is a solution
				if np.all(self.grid != 0):
					yield self.grid.copy()
				# If current grid is not impossible
				elif np.all(self.counter[..., 0] != 0):
					noptions = self.counter[..., 0] + (self.grid != 0) * self.N

					## Choose guessed cell
					key = np.unravel_index(np.argmin(noptions), self.grid.shape)

					if shuffle:
						keys = np.argwhere(noptions == noptions[key])
						key = tuple(keys[np.random.randint(len(keys))])

					guesses.append((key, self.options(key)))
			else:
				backprop = False

			# If currently guessing
			if guesses:
				key, options = guesses.pop()

				## If there is still value options
				if options:
					self[key] = options.pop()
					guesses.append((key, options))
				else:
					self[key] = 0
					backprop = True
			else:
				break

	def unsolve(self, shuffle=True):
		'''Return a minimal subgrid that respects uniqueness.'''

		subgrid = Sudoku(n=self.n, d=self.d)

		# Order according to number of value options
		noptions = self.counter[..., 0]

		if shuffle:
			order = np.random.permutation(noptions.size)
			keys = order[np.argsort(-noptions.ravel()[order])]
		else:
			keys = np.argsort(-noptions.ravel())

		# Empty cells
		for key in zip(*np.unravel_index(keys, self.grid.shape)):
			if self.empty(key):
				continue

			if noptions[key] > 1:
				np.copyto(subgrid.grid, self.grid)
				np.copyto(subgrid.counter, self.counter)

				## Empty cell & add value exception
				subgrid[key] = 0
				subgrid.counter[key][self[key]] = 1

				## If there exists other solutions
				for _ in subgrid.solve():
					break
				else:
					self[key] = 0
			else:
				self[key] = 0

		return self.grid.copy()


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
		sudoku = Sudoku(n=3, d=2)
	else:
		sudoku = Sudoku(grid=np.loadtxt(args.file, dtype=int, delimiter=args.sep))

	# Solutions
	for solution in sudoku.solve():
		np.savetxt(
			sys.stdout,
			Sudoku(grid=solution).unsolve() if args.unsolved else solution,
			'%0{}d'.format(int(np.log10(sudoku.N)) + 1),
			delimiter=args.sep
		)

		if args.number > 1:
			args.number -= 1
			print(end='\n')
		else:
			break
