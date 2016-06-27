"""Make some figures based on a model's generalisation performance"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import train_bilinear
import data


flags = tf.app.flags

# we will already have the params from train_bilinear which control the model,
# but we need some here as well such as where the validation files are etc.
flags.DEFINE_string('valid_set', 'validation_labels.txt',
                    'The path to the file that contains labels defining the '
                    'items that the model did not see during training.')
flags.DEFINE_string('output_dir', '.', 'where to put the new images we are '
                    'making')
FLAGS = flags.FLAGS


def load_validation_labels(path):
    """Load labels from a file."""
    with open(path) as v_file:
        all_labels = v_file.read().rstrip()
        all_labels = [line.split(',') for line in all_labels.split('\n')[1:]]
        all_labels = [(int(label[1]), int(label[0]))
                      for label in all_labels]
    return all_labels


def main(_):
    """Load up a model, get its validation set and make them all into
    pictures"""
    # keep in mind this is backwards, this is in fact the training set.
    # whoops.
    pairs = load_validation_labels(FLAGS.valid_set)
    images, clabel, slabel = data.get_tf_images(52, pairs=pairs)
    sembed, cembed = train_bilinear.embed(slabel, clabel)
    # have data, get model
    with tf.variable_scope('model'):
        model_out = train_bilinear.get_bilinear_output(
            sembed, cembed, summarise=False)


if __name__ == '__main__':
    tf.app.run()
