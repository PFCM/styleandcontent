"""Read the data"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import string

import numpy as np
from PIL import Image


def get_images(directory, ids):
    """Loads all the png images in a directory.
    Looks up their names in ids"""
    files = [f
             for f in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, f))]
    files = [f for f in files if f.endswith('.png')]
    
    data = []
    labels = []
    for fname in files:
        img = Image.open(os.path.join(directory, fname))
        img.load()
        data.append(np.asarray(img, dtype=np.uint8)[..., 0].reshape((1, 1024)))
        labels.append(ids[fname.split('.')[0]])
    return np.vstack(data), labels


def get_all_characters():
    """finds everything.

    Returns:
        data, charlabels, fontlabels, charids, fontids
    """
    outer_dir = os.path.dirname(__file__)
    all_dirs = [f for f in os.listdir(outer_dir)
                if not os.path.isfile(os.path.join(outer_dir, f))]
    char_ids = {char: i for i, char in enumerate(string.ascii_letters)}
    font_ids = {font: i for i, font in enumerate(all_dirs)}
    all_data = []
    all_charlabels = []
    all_fontlabels = []
    for font_dir in all_dirs:
        data, clabels = get_images(os.path.join(outer_dir, font_dir),
                                   char_ids)
        all_data.append(data)
        all_charlabels += clabels
        all_fontlabels += [font_ids[font_dir]] * len(clabels)
    return np.vstack(all_data), all_charlabels, all_fontlabels, char_ids, font_ids


if __name__ == '__main__':
    data, clabels, flabels, cids, fids = get_all_characters()
    print(data.shape)
    print(len(clabels), len(flabels))
    print(cids)
    print(fids)
