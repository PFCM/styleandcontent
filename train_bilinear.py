"""See if we can just learn to reproduce the data using a bilinear model.

What we will do is:
- represent style and content by one-hots
- multiply them by an embedding matrix of fixed dimensionality to get
  appropriate vectors.
- push the vectors through a bilinear product with a 3 tensor to produce an
  image vector.

The task then is to learn the tensor and the embedding matrices at the same
time. We can test by holding out some examples and seeing how it is able to
reproduce things it has not seen (novel style/content combinations from styles
and contents it has seen will be straightforward, but we might have to figure
out a bit of a process for entirely new styles/contents).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import datetime

import numpy as np
import tensorflow as tf

import progressbar

import data
import mrnn


flags = tf.app.flags

# learning params
flags.DEFINE_integer('max_epochs', 1000, 'how long to train for')
flags.DEFINE_float('learning_rate', 0.1, 'learning rate for gradient descent')
flags.DEFINE_integer('batch_size', 10, 'how many to do at once.')

# model params
flags.DEFINE_integer('style_embedding_dim', 30, 'how big to make the style '
                     'embedding vectors')
flags.DEFINE_integer('content_embedding_dim', 30, 'how big to make the content'
                     ' embeddings')
flags.DEFINE_string('decomposition', 'tt', 'which tensor decomposition to use.'
                    ' one of `tt`, `cp`, `sparse` or `none`. Be careful with '
                    '`none`, it might get very large indeed.')
flags.DEFINE_integer('rank1', 10, 'the rank of the tensor. For the CP '
                     'decomposition this is the only rank (the rank2 flag is '
                     'ignored). For tensor-train it is the first rank and in '
                     'this model corresponds to the dimension that the style '
                     'embeddings are projected into')
flags.DEFINE_integer('rank2', 10, 'For tensor-train, this is the second '
                     'tt-rank and in this bilinear model corresponds to the '
                     'dimensionality that the style embeddings are projected '
                     'into')
flags.DEFINE_float('sparsity', 0.15, 'The fraction/number of non-zero elements'
                   ' if a sparse tensor is used. If > 0 then it is assumed to '
                   ' a count, otherwise a fraction.')

# admin stuff
flags.DEFINE_string('logdir', 'logs', 'Where to save the logs of the runs')
flags.DEFINE_string('savepath', 'models', 'where to save the models')

FLAGS = flags.FLAGS


def _get_tt_model(style_input, content_input):
    """Gets a TT tensor and returns the product of it with the two input
    vectors."""
    tensor = mrnn.tensor_ops.get_tt_3_tensor(
        [FLAGS.style_embedding_dim, data.IMAGE_SIZE**2,
         FLAGS.content_embedding_dim],
        [FLAGS.rank1, FLAGS.rank2],
        trainable=True,
        name='TT_model')
    for mat in tensor[:3]:  # add summaries of the three matrices
        tf.histogram_summary(mat.op.name + '_hist', mat)
    # and do the product
    return mrnn.tensor_ops.bilinear_product_tt_3(
        style_input, tensor, content_input)


def _get_cp_model(style_input, content_input):
    """Gets a cp tensor and returns the product of it with the two input
    vectors."""
    tensor = mrnn.tensor_ops.get_cp_tensor(
        [FLAGS.style_embedding_dim, data.IMAGE_SIZE**2,
         FLAGS.content_embedding_dim],
        FLAGS.rank1,
        trainable=True,
        name='CP_model')
    for mat in tensor:  # add summaries of the three matrices
        tf.histogram_summary(mat.op.name + '_hist', mat)
    # and do the product
    return mrnn.tensor_ops.bilinear_product_cp(
        style_input, tensor, content_input)


# TODO(pfcm) sparse subset for quicker test.
def _get_sparse_model(style_input, content_input):
    """Gets a sparse tensor and returns the product of it with the two input
    vectors."""
    tensor = mrnn.tensor_ops.get_sparse_tensor(
        [FLAGS.style_embedding_dim, data.IMAGE_SIZE**2,
         FLAGS.content_embedding_dim],
        FLAGS.sparsity,
        trainable=True,
        name='sparse_model')
    for mat in tensor:  # add summaries of the three matrices
        tf.histogram_summary(mat.op.name + '_hist', mat)
    # and do the product
    return mrnn.tensor_ops.sparse(
        style_input, tensor, content_input, data.IMAGE_SIZE**2)


def _get_full_model(style_input, content_input):
    """Probably impractical, but seems like it might be worth checking?"""
    pass


def get_bilinear_output(style_input, content_input):
    """Gets the bilinear product part of the model given the embedded inputs.
    Checks FLAGS for the appropriate decomposition, size etc.

    Args:
        style_input: `[batch_size, style_embedding_dim]` input tensor.
        content_input: `[batch_size, content_embedding_dim]` input tensor.

    Returns:
        `[batch_size, image_size]` output tensor.
    """
    if FLAGS.decomposition == 'tt':
        return _get_tt_model(style_input, content_input)
    if FLAGS.decomposition == 'cp':
        return _get_cp_model(style_input, content_input)
    if flags.decomposition == 'sparse':
        return _get_sparse_model(style_input, content_input)
    if flags.decomposition == 'none':
        return _get_full_model(style_input, content_input)
    raise ValueError('Unkown decomposition: {}'.format(FLAGS.decomposition))


def embed(style_labels, content_labels):
    """Get embedding matrices and return lookup tensors.
    Looks in FLAGS for the size of the embeddings.

    Args:
        style_labels: `[batch_size, num_styles]` int tensor of styles.
        content_labels: `[batch_size, num_contents]` int tensor of content ids.

    Returns:
        (style, content): pair of `[batch_size, embedding_dim]` tensors with
            lookups.
    """
    # this is pretty straightforward
    with tf.variable_scope('style_embedding'):
        style_embedding = tf.get_variable(
            'style_embedding_matrix',
            [data.NUM_STYLE_LABELS, FLAGS.style_embedding_dim],
            dtype=tf.float32,
            trainable=True)
        styles = tf.nn.embedding_lookup(style_embedding,
                                        style_labels)

    with tf.variable_scope('content_embedding'):
        content_embedding = tf.get_variable(
            'content_embedding_matrix',
            [data.NUM_CONTENT_LABELS, FLAGS.content_embedding_dim],
            dtype=tf.float32,
            trainable=True)
        contents = tf.nn.embedding_lookup(content_embedding,
                                          content_labels)
    return styles, contents


def mse(x, y):
    """Returns mean squared error"""
    return tf.reduce_mean(tf.squared_difference(x, y))


def get_train_op(loss_op, global_step):
    """Gets a training op. Looks in flags for the parameters of the
    optimizer and assumes we want to train everything in the
    TRAINABLE_VARIABLES collection."""
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    return opt.minimize(loss_op, global_step=global_step)


def main(_):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s](%(funcName)s-%(levelname)s) %(message)s')
    # get the data batch tensors
    # for now, for testing, we will just be training on the whole lot
    # and see if it can learn anything
    image, clabel, slabel = data.get_tf_images(FLAGS.batch_size,
                                               num_epochs=FLAGS.max_epochs)
    tf.image_summary('input', tf.reshape(
        image, [-1, data.IMAGE_SIZE, data.IMAGE_SIZE, 1]))
    s_embed, c_embed = embed(slabel, clabel)
    with tf.variable_scope('model'):
        train_out = get_bilinear_output(s_embed, c_embed)
        tf.image_summary(
            'model_output',
            tf.reshape(
                train_out,
                [FLAGS.batch_size, data.IMAGE_SIZE, data.IMAGE_SIZE, 1]))
    # get some training stuff
    with tf.variable_scope('training'):
        loss_op = mse(train_out, image)
        # add any regularisation losses
        tf.scalar_summary('loss', loss_op)
        global_step = tf.Variable(0, trainable=False, name='global_step')
        train_op = get_train_op(loss_op, global_step)

    # can't be bothered saving yet
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    # do want to log progress
    if os.path.exists(FLAGS.logdir) and os.path.isdir(FLAGS.logdir):
        logdir = os.path.join(
            FLAGS.logdir,
            datetime.datetime.now().strftime('%Y-%m-%d--%H%M%S'))
    else:
        logdir = FLAGS.logdir
    s_writer = tf.train.SummaryWriter(logdir, graph=sess.graph)
    all_summaries = tf.merge_all_summaries()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    total_steps = FLAGS.max_epochs * data.NUM_STYLE_LABELS * \
        data.NUM_CONTENT_LABELS // FLAGS.batch_size
    print('About to train for {} steps.'.format(total_steps))
    with sess.as_default():
        try:
            bar = progressbar.ProgressBar(
                max_value=total_steps,
                widgets=[
                    '[', progressbar.Percentage(), '] ',
                    '(๑•﹏•)⋆⁑*⋆',
                    progressbar.Bar(left='',
                                    fill=' ',
                                    marker='⋆'),
                    '⠅', progressbar.DynamicMessage('loss'), '⠅ ',
                    '[', progressbar.AdaptiveETA(), ']'],
                redirect_stdout=True)
            bar.start()
            while not coord.should_stop():

                loss, _ = sess.run([loss_op, train_op])

                step = global_step.eval()
                bar.update(step, loss=loss)
                if step % 10 == 0:
                    summs = sess.run(all_summaries)
                    s_writer.add_summary(summs, global_step=step)

        except tf.errors.OutOfRangeError:
            bar.finish()
            print('Done.')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    tf.app.run()
