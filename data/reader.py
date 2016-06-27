"""Read the data"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import string
import logging

import numpy as np
from PIL import Image

import tensorflow as tf


IMAGE_SIZE = 32
# need to keep track of these
NUM_STYLE_LABELS = 10
NUM_CONTENT_LABELS = len(string.ascii_letters)


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
        dirs = get_dirs()
    chars = [str(ord(c)) for c in string.ascii_letters]
    char_ids = {char: i for i, char in enumerate(chars)}
    font_ids = {font: i for i, font in enumerate(sorted(dirs))}
    return char_ids, font_ids


def get_characters(chars=string.ascii_letters, pairs=None):
    """finds everything. or not. If present pairs should be a list of
    (charlabel, fontlabel) tuples (where the labels come from get_vocabs).

    Returns:
        data, charlabels, fontlabels, charids, fontids
    """
    chars = [str(ord(c)) for c in chars]
    all_dirs = get_dirs()
    char_ids, font_ids = get_vocabs(all_dirs)
    all_data = []
    all_charlabels = []
    all_fontlabels = []
    for font_dir in all_dirs:
        data, clabels = get_images(font_dir,
                                   char_ids,
                                   chars=chars)
        if pairs:
            labels = [(clabel, font_ids[font_dir]) for clabel in clabels]
            data = [item for item, label in zip(data, labels)
                    if label not in pairs]
            clabels = [clabel for clabel, label in zip(clabels, labels)
                       if label not in pairs]
        all_data.append(data)
        all_charlabels += clabels
        all_fontlabels += [font_ids[font_dir]] * len(clabels)
    all_charlabels = np.array(all_charlabels, dtype=np.int32)
    all_fontlabels = np.array(all_fontlabels, dtype=np.int32)
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


def _batch_data(images, clabels, slabels, batch_size, num_epochs):
    input_images = tf.Variable(images, trainable=False)
    input_clabels = tf.Variable(clabels, trainable=False)
    input_slabels = tf.Variable(slabels, trainable=False)

    image, clabel, slabel = tf.train.slice_input_producer(
        [input_images, input_clabels, input_slabels],
        num_epochs=num_epochs,
        shuffle=True, capacity=batch_size*3)

    # get the images ready
    float_images = tf.cast(image, tf.float32)
    float_images = (float_images / 127.0) - 1.0

    return tf.train.batch(
        [float_images, clabel, slabel], batch_size=batch_size)


def get_tf_images(batch_size, chars=string.ascii_letters,
                  pairs=None, min_after_dequeue=100, num_epochs=None,
                  validation=0.0, write_split=True):
    """Probably the best move is still to just grab everything and then slice
    it (we aren't dealing with much data at all). Writes labels of the
    validation data to a given text file so that you can reuse them"""
    all_data, all_clabels, all_slabels, cvocab, svocab = get_characters(chars,
                                                                        pairs)
    logging.info('Got %d images', all_data.shape[0])
    # be careful not to save these
    if validation <= 0.0:
        return _batch_data(all_data, all_clabels, all_slabels, batch_size,
                           num_epochs)
    else:
        # we will have to do it twice
        idces = np.arange(all_data.shape[0], dtype=np.int32)
        np.random.shuffle(idces)
        num_valid = int(all_data.shape[0] * validation)
        logging.info('Validating with %d images', num_valid)
        valid_idces = idces[:num_valid]
        train_idces = idces[num_valid:]

        if write_split:
            with open('train_labels.txt', 'w') as f:
                f.write('style,content\n')
                for s, c in zip(all_slabels[train_idces],
                                all_clabels[train_idces]):
                    f.write('{},{}\n'.format(s, c))

        train_batches = _batch_data(all_data[train_idces, :],
                                    all_clabels[train_idces],
                                    all_slabels[train_idces],
                                    batch_size,
                                    num_epochs)
        valid_batches = _batch_data(all_data[valid_idces, :],
                                    all_clabels[valid_idces],
                                    all_slabels[valid_idces],
                                    batch_size,
                                    None)
        return train_batches, valid_batches


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
