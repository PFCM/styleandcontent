"""Read the data"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import string
import itertools
import logging

import numpy as np
from PIL import Image

import tensorflow as tf


IMAGE_SIZE = 32
# need to keep track of these
NUM_STYLE_LABELS = 10
NUM_CONTENT_LABELS = len(string.ascii_uppercase)


def get_images(directory, ids, chars):
    """Loads all the png images in a directory (with names in `chars`).
    Looks up their names in ids"""
    files = [f for f in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, f))]
    files = [f for f in files
             if f.endswith('.png') and f.split('.')[0] in chars]

    data = []
    labels = []
    for fname in files:
        img = Image.open(os.path.join(directory, fname))
        img.load()
        data.append(np.asarray(img, dtype=np.uint8)[..., 0].reshape((1, 1024)))
        labels.append(ids[fname.split('.')[0]])
    return np.vstack(data), labels


def get_dirs():
    """Returns a list of all of the folders with data in them"""
    outer_dir = os.path.dirname(__file__)
    logging.info('looking for data in %s', outer_dir)
    all_dirs = [os.path.join(outer_dir, f)
                for f in os.listdir(outer_dir)
                if not os.path.isfile(os.path.join(outer_dir, f)) and
                not f.startswith('_')]
    logging.info('  found %d directories to check', len(all_dirs))
    return all_dirs


def get_vocabs(dirs=None):
    """Gets the dictionaries mapping labels -> ints"""
    if not dirs:
        dirs = get_dirs

    char_ids = {char: i for i, char in enumerate(string.ascii_uppercase)}
    font_ids = {font: i for i, font in enumerate(sorted(dirs))}
    return char_ids, font_ids


def get_characters(chars=string.ascii_letters):
    """finds everything.

    Returns:
        data, charlabels, fontlabels, charids, fontids
    """
    all_dirs = get_dirs()
    char_ids, font_ids = get_vocabs(all_dirs)
    all_data = []
    all_charlabels = []
    all_fontlabels = []
    for font_dir in all_dirs:
        data, clabels = get_images(font_dir,
                                   char_ids,
                                   chars=chars)
        all_data.append(data)
        all_charlabels += clabels
        all_fontlabels += [font_ids[font_dir]] * len(clabels)
    NUM_STYLE_LABELS = len(all_fontlabels)
    return (np.vstack(all_data),
            all_charlabels, all_fontlabels, char_ids, font_ids)


class _path_to_labels(object):

    def __init__(self, style_vocab, content_vocab):
        self.style_vocab = style_vocab
        self.content_vocab = content_vocab

    def __call__(self, paths):
        splits = np.char.split(paths, sep=os.path.sep)
        style_ids = np.array([self.style_vocab[s[1]] for s in splits])
        content_ids = np.array([self.content_vocab[s[-1][0:1]]
                                for s in splits])
        return np.array([style_ids, content_ids])


def get_tf_images(batch_size, chars=string.ascii_uppercase,
                  min_after_dequeue=100, num_epochs=None):
    """Probably the best move is still to just grab everything and then slice
    it (we aren't dealing with much data at all)."""
    all_data, all_clabels, all_slabels, cvocab, svocab = get_characters(chars)
    logging.info('Got %d images', all_data.shape[0])
    # be careful not to save these
    input_images = tf.Variable(all_data, trainable=False)
    input_clabels = tf.Variable(all_clabels, trainable=False)
    input_slabels = tf.Variable(all_slabels, trainable=False)

    image, clabel, slabel = tf.train.slice_input_producer(
        [input_images, input_clabels, input_slabels], num_epochs=num_epochs,
        shuffle=True, capacity=32)

    # get the images ready
    float_images = tf.cast(image, tf.float32)
    float_images = (float_images / 127.0) - 1.0

    return tf.train.batch(
        [float_images, clabel, slabel], batch_size=batch_size)


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    img, content, style = get_tf_images(1, num_epochs=1)
    sess.run(tf.initialize_all_variables())

    # start the threads only after init
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        count = 0
        while not coord.should_stop():
            a, b, c = sess.run([img, content, style])
            print(b, c)
            count += 1
    except tf.errors.OutOfRangeError:
        print('{} in total'.format(count))
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
