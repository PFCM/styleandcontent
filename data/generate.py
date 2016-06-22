"""Script to generate the data from font files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import string

import numpy as np

from PIL import Image, ImageFont, ImageDraw


def main():
    filename = sys.argv[1]
    print('Loading font from {}'.format(filename))
    font = ImageFont.truetype(filename, 34)  # TODO size
    output_folder = os.path.join(
        os.path.dirname(__file__),
        os.path.basename(filename).split('.')[0])
    os.makedirs(output_folder, exist_ok=True)
    print('~~saving to {}'.format(output_folder))
    for char in string.ascii_letters:
        # CASE SENSITIVITY BUG
        # should come up with names that don't rely on cases
        # and regenerate
        image = Image.new('RGB', (32, 32))
        draw = ImageDraw.Draw(image)
        w, h = font.getsize(char)
        draw.text(((32-w)/2, (28-h)/2), char, font=font)
        image.save(os.path.join(output_folder, char)+'.png', 'PNG')
    

if __name__ == '__main__':
    main()
