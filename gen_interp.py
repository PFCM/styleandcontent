"""make some pics"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np
import tensorflow as tf

from train_bilinear import get_bilinear_output, embed
import data

flags = tf.app.flags
flags.DEFINE_string('content_start', 'J', 'first letter for the content '
                    'interpolation')
flags.DEFINE_string('content_end', 'w', 'last letter for the content, also'
                    ' fixed content when interpolating the other guy')
flags.DEFINE_string('style_start', 'Georgiai', 'first style (also fixed style)')
flags.DEFINE_string('style_end', 'VeraMono', 'the last style')
flags.DEFINE_string('loadpath', '.', 'where to try and load a model from. '
                    'If it is a directory we will find the latest checkpoint '
                    'otherwise we will try and load it as a checkpoint itself')
flags.DEFINE_integer('interp_steps', 25, 'how many figures to make')
flags.DEFINE_string('outpath', '.', 'where to write the pics')
FLAGS = flags.FLAGS


def interpolate(start, end, steps):
    """interpolate linearly between two vectors. Returns the lot as
    a big matrix"""
    increment = (end - start) / steps
    diffs = increment * np.arange(steps)[:, np.newaxis]
    return diffs + start.flatten()


def get_image_encoders(data):
    """Get a tensor to turn a batch of pixels into a batch of png encoded
    bytes."""
    batch_size = data.get_shape().as_list()[0]
    # normalise the images
    # this assumes they are all positive, this could be drastically wrong
    int_data = tf.saturate_cast(data / tf.reduce_max(data, 1, keep_dims=True) *
                                255.0,
                                tf.uint8)
    encoders = [tf.image.encode_png(tf.reshape(int_data[i, :], [32, 32, 1]))
                for i in range(batch_size)]
    return tf.pack(encoders)


def write_images(data):
    """gets a numpy array of `object`, assumed to be bytes and writes them
    to disk wth a .png extension"""
    if not os.path.exists(FLAGS.outpath):
        os.makedirs(FLAGS.outpath)
    for i, image_bytes in enumerate(data):
        savepath = os.path.join(FLAGS.outpath, 'output_{}.png'.format(i))
        with open(savepath, 'wb') as fout:
            fout.write(image_bytes)


def main(_):
    """Load up a model, restore it from file, run it on some carefully
    generated input, print the output"""
    logging.basicConfig(level=logging.INFO)

    style_placeholder = tf.placeholder(
        tf.float32,
        shape=[FLAGS.interp_steps*2, FLAGS.style_embedding_dim])
    content_placeholder = tf.placeholder(
        tf.float32,
        shape=[FLAGS.interp_steps*2, FLAGS.content_embedding_dim])
    with tf.variable_scope('model'):
        model_out = get_bilinear_output(style_placeholder,
                                        content_placeholder)
    image_batch = get_image_encoders(model_out)
    s_embed_placeholder = tf.placeholder(tf.int32, shape=[1])
    c_embed_placeholder = tf.placeholder(tf.int32, shape=[1])
    s_embedding, c_embedding = embed(s_embed_placeholder, c_embed_placeholder)
    # let's load it up
    saver = tf.train.Saver(tf.trainable_variables())
    if os.path.isdir(FLAGS.loadpath):
        logging.info('Looking for latest checkpoint in %s',
                     FLAGS.loadpath)
        loadpath = tf.train.latest_checkpoint(FLAGS.loadpath)
        logging.info('Got %s', loadpath)
    else:
        loadpath = FLAGS.loadpath
        logging.info('Trying to load %s', loadpath)
    content_vocab, style_vocab = data.get_vocabs()
    sess = tf.Session()
    saver.restore(sess, loadpath)
    with sess.as_default():
        start_style_label = os.path.abspath(os.path.join(
            'data', FLAGS.style_start))
        end_style_label = os.path.abspath(os.path.join(
            'data', FLAGS.style_end))

        style_start = sess.run(
            s_embedding,
            {s_embed_placeholder: [style_vocab[start_style_label]]})
        style_end = sess.run(
            s_embedding,
            {s_embed_placeholder: [style_vocab[end_style_label]]})
        style_batch = interpolate(style_start, style_end, FLAGS.interp_steps)

        # and get the content
        start_content_label = str(ord(FLAGS.content_start))
        end_content_label = str(ord(FLAGS.content_end))
        content_start = sess.run(
            c_embedding,
            {c_embed_placeholder: [content_vocab[start_content_label]]})
        content_end = sess.run(
            c_embedding,
            {c_embed_placeholder: [content_vocab[end_content_label]]})
        content_batch = interpolate(content_start, content_end,
                                    FLAGS.interp_steps)
        # content comes first
        content_batch = np.vstack(
            (content_batch,
             np.array([content_end[0] for _ in range(FLAGS.interp_steps)])))
        style_batch = np.vstack(
            (np.array([style_start[0] for _ in range(FLAGS.interp_steps)]),
             style_batch))
        # should be good to get some pics
        pics = sess.run(image_batch,
                        {style_placeholder: style_batch,
                         content_placeholder: content_batch})
        write_images(pics)


if __name__ == '__main__':
    tf.app.run()
