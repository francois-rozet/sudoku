#!/usr/bin/env python
# -*- coding: utf-8 -*-

#############
# Libraries #
#############

import os
import numpy as np
from numpy.random import choice, rand
import cairo
import img2pdf


##############
# Parameters #
##############

SIZE = 3
N_NUMBER = 54
N_GRID = 400

FONT = [
    'Arial',
    'Cambria',
    'Comic Sans MS',
    'Courier New',
    'Ink Free',
    'Lato',
    'Segoe Print'
]

DESTINATION = 'images/generated/'
FILENAME = 'sudoku_{:04d}'

#############
# Functions #
#############

def gen(n, p):
    grid = np.zeros((n ** 2, n ** 2), dtype=int)

    for ind in np.random.choice(n ** 2, (p, 2)):
        grid[ind[0], ind[1]] = np.random.randint(1, n ** 2 + 1)

    return grid

def draw(filename, grid, size=1024):
    filename += '.png'

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
        ctx.rotate((rand() - 1 / 2) * np.pi / 18)
        ctx.rectangle(
            rand() * 2 * start,
            rand() * 2 * start,
            1.05 * (end - start),
            1.05 * (end - start)
        )
        ctx.set_source_rgb(0.6 + rand() * 0.4, 0.6 + rand() * 0.4, 0.6 + rand() * 0.4)
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
            FONT[choice(len(FONT))],
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
    # Prevent randomness
    np.random.seed(0)

    img_list = []

    for i in range(1, N_GRID + 1):
        # Generation
        grid = gen(SIZE, N_NUMBER)

        # mkdir -p DESTINATION
        os.makedirs(DESTINATION, exist_ok=True)

        # Save
        filename = DESTINATION + FILENAME.format(i)
        np.savetxt(filename + '.dat', grid, fmt='%d')
        img_list.append(draw(filename, grid))

    with open(DESTINATION + 'sudoku_all.pdf', 'wb') as f:
        f.write(img2pdf.convert(img_list))
