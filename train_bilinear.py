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
flags.DEFINE_float('l2_reg', 0.001, 'amount of l2 regularisation (weight '
                   'decay)')
flags.DEFINE_integer('early_stop', 3, 'if the validation error is worse than '
                     'the last `early_stop` checks, then quit early.')

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
flags.DEFINE_bool('conv', False, 'Whether to add a small conv layer on top')

# admin stuff
flags.DEFINE_string('logdir', 'logs', 'Where to save the logs of the runs')
flags.DEFINE_string('savepath', '', 'where to save the models. If empty, no '
                    'saving.')
flags.DEFINE_integer('save_every', 1000, 'how many batches after which to '
                     'save')
flags.DEFINE_float('validation', 0.1, 'how much of the data to split off for '
                   'validation')
FLAGS = flags.FLAGS


def _get_tt_model(style_input, content_input, summarise):
    """Gets a TT tensor and returns the product of it with the two input
    vectors."""
    tensor = mrnn.tensor_ops.get_tt_3_tensor(
        [FLAGS.style_embedding_dim, data.IMAGE_SIZE**2,
         FLAGS.content_embedding_dim],
        [FLAGS.rank1, FLAGS.rank2],
        trainable=True,
        name='TT_model')
    if summarise:
        for mat in tensor[:3]:  # add summaries of the three matrices
            tf.histogram_summary(mat.op.name + '_hist', mat)
    # and do the product
    return mrnn.tensor_ops.bilinear_product_tt_3(
        style_input, tensor, content_input)


def _get_cp_model(style_input, content_input, summarise):
    """Gets a cp tensor and returns the product of it with the two input
    vectors."""
    tensor = mrnn.tensor_ops.get_cp_tensor(
        [FLAGS.style_embedding_dim, data.IMAGE_SIZE**2,
         FLAGS.content_embedding_dim],
        FLAGS.rank1,
        trainable=True,
        name='CP_model')
    if summarise:
        for mat in tensor:  # add summaries of the three matrices
            tf.histogram_summary(mat.op.name + '_hist', mat)
    # and do the product
    return mrnn.tensor_ops.bilinear_product_cp(
        style_input, tensor, content_input)


# TODO(pfcm) sparse subset for quicker test.
def _get_sparse_model(style_input, content_input, summarise):
    """Gets a sparse tensor and returns the product of it with the two input
    vectors."""
    tensor = mrnn.tensor_ops.get_sparse_tensor(
        [FLAGS.style_embedding_dim, data.IMAGE_SIZE**2,
         FLAGS.content_embedding_dim],
        FLAGS.sparsity,
        name='sparse_model')
    # and do the product
    return mrnn.tensor_ops.bilinear_product_sparse(
        style_input, tensor, content_input, data.IMAGE_SIZE**2)


def _get_full_model(style_input, content_input, summarise):
    """Probably impractical, but seems like it might be worth checking?"""
    pass


def get_bilinear_output(style_input, content_input, summarise=True):
    """Gets the bilinear product part of the model given the embedded inputs.
    Checks FLAGS for the appropriate decomposition, size etc.

    Args:
        style_input: `[batch_size, style_embedding_dim]` input tensor.
        content_input: `[batch_size, content_embedding_dim]` input tensor.
        summarise: whether to add summaries to the graph.

    Returns:
        `[batch_size, image_size]` output tensor.
    """
    if FLAGS.decomposition == 'tt':
        layer = _get_tt_model(style_input, content_input, summarise)
    elif FLAGS.decomposition == 'cp':
        layer = _get_cp_model(style_input, content_input, summarise)
    elif FLAGS.decomposition == 'sparse':
        layer = _get_sparse_model(style_input, content_input, summarise)
    elif FLAGS.decomposition == 'none':
        layer = _get_full_model(style_input, content_input, summarise)
    else:
        raise ValueError(
            'Unkown decomposition: {}'.format(FLAGS.decomposition))
    # maybe for kicks
    if FLAGS.conv:
        batch_size = style_input.get_shape().as_list()[0]
        layer = tf.reshape(tf.nn.relu(layer), [batch_size, 8, 8, 16])
        filters = tf.get_variable('conv_filters', [5, 5, 1, 16])
        layer = tf.nn.conv2d_transpose(layer, filters, [batch_size, 32, 32, 1],
                                       [1, 4, 4, 1])
        return tf.reshape(tf.nn.elu(layer), [batch_size, 1024])
    else:
        return tf.nn.elu(layer)


def embed(style_labels, content_labels, reuse=False):
    """Get embedding matrices and return lookup tensors.
    Looks in FLAGS for the size of the embeddings.

    Args:
        style_labels: `[batch_size]` int tensor of styles.
        content_labels: `[batch_size]` int tensor of content ids.
        reuse: whether to try and get variables that already exist.

    Returns:
        (style, content): pair of `[batch_size, embedding_dim]` tensors with
            lookups.
    """
    # this is pretty straightforward
    with tf.variable_scope('style_embedding', reuse=reuse):
        style_embedding = tf.get_variable(
            'style_embedding_matrix',
            [data.NUM_STYLE_LABELS, FLAGS.style_embedding_dim],
            dtype=tf.float32,
            trainable=True)
        # style_embedding = tf.nn.l2_normalize(style_embedding, 1)
        styles = tf.nn.embedding_lookup(style_embedding,
                                        style_labels)

    with tf.variable_scope('content_embedding', reuse=reuse):
        content_embedding = tf.get_variable(
            'content_embedding_matrix',
            [data.NUM_CONTENT_LABELS, FLAGS.content_embedding_dim],
            dtype=tf.float32,
            trainable=True)
        # content_embedding = tf.nn.l2_normalize(content_embedding, 1)
        contents = tf.nn.embedding_lookup(content_embedding,
                                          content_labels)
    return styles, contents


def l2_regulariser(amount):
    """Returns a function which returns half the sum of squares of the given
    variable, times amount."""

    def _l2_reg(var):
        return amount * tf.nn.l2_loss(var)

    return _l2_reg


def l1_regulariser(amount):
    """sum of absolute values"""

    def _l1_reg(var):
        return amount * tf.reduce_sum(tf.abs(var))

    return _l1_reg


def mse(x, y):
    """Returns mean squared error"""
    return tf.reduce_mean(tf.squared_difference(x, y))


def get_train_op(loss_op, global_step):
    """Gets a training op. Looks in flags for the parameters of the
    optimizer and assumes we want to train everything in the
    TRAINABLE_VARIABLES collection. Also adds all of REGULARIZATION_LOSSES
    to loss_op before minimising, so don't do that outside."""
    loss_op = tf.add_n([loss_op] +
                       tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    return opt.minimize(loss_op, global_step=global_step)


def main(_):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s](%(funcName)s-%(levelname)s) %(message)s')
    # get the data batch tensors
    # for now, for testing, we will just be training on the whole lot
    # and see if it can learn anything
    if FLAGS.validation != 0.0:
        train, valid = data.get_tf_images(FLAGS.batch_size,
                                          num_epochs=FLAGS.max_epochs,
                                          validation=FLAGS.validation)
        t_image, t_clabel, t_slabel = train
        v_image, v_clabel, v_slabel = valid
    else:
        t_image, t_clabel, t_slabel = data.get_tf_images(
            FLAGS.batch_size,
            num_epochs=FLAGS.max_epochs)

    tf.image_summary('train_input', tf.reshape(
        t_image, [-1, data.IMAGE_SIZE, data.IMAGE_SIZE, 1]))
    t_sembed, t_cembed = embed(t_slabel, t_clabel)
    if FLAGS.validation != 0.0:
        tf.image_summary('valid_input', tf.reshape(
            v_image, [-1, data.IMAGE_SIZE, data.IMAGE_SIZE, 1]))
        v_sembed, v_cembed = embed(v_slabel, v_clabel, reuse=True)
    with tf.variable_scope('model',
                           regularizer=l2_regulariser(FLAGS.l2_reg)) as scope:
        train_out = get_bilinear_output(t_sembed, t_cembed)
        tf.image_summary(
            'model_output',
            tf.reshape(
                train_out,
                [FLAGS.batch_size, data.IMAGE_SIZE, data.IMAGE_SIZE, 1]))
        scope.reuse_variables()
        if FLAGS.validation != 0.0:
            valid_out = get_bilinear_output(v_sembed, v_cembed,
                                            summarise=False)
            tf.image_summary(
                'validation_output',
                tf.reshape(
                    valid_out,
                    [FLAGS.batch_size, data.IMAGE_SIZE, data.IMAGE_SIZE, 1]))
    # get some training stuff
    with tf.variable_scope('training'):
        loss_op = mse(train_out, t_image)
        # add any regularisation losses
        tf.scalar_summary('loss', loss_op)
        global_step = tf.Variable(0, trainable=False, name='global_step')
        train_op = get_train_op(loss_op, global_step)

    if FLAGS.validation != 0.0:
        with tf.variable_scope('validation'):
            valid_loss = mse(valid_out, v_image)
            tf.scalar_summary('valid_loss', valid_loss)

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
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
            valid_losses = []
            while not coord.should_stop():

                loss, _ = sess.run([loss_op, train_op])

                step = global_step.eval()
                bar.update(step, loss=loss)
                if step % 10 == 0:
                    summs = sess.run(all_summaries)
                    s_writer.add_summary(summs, global_step=step)
                if step % FLAGS.save_every == 0:
                    if FLAGS.validation != 0.0:
                        vloss, = sess.run([valid_loss])
                        print('validation loss (one batch): {}'.format(vloss))
                        if FLAGS.early_stop:
                            if (len(valid_losses) > FLAGS.early_stop and
                              vloss >= max(valid_losses[-FLAGS.early_stop:])):
                                print('Not so good, stopping early.')
                                break
                            else:
                                valid_losses.append(vloss)
                    saver.save(sess, FLAGS.savepath, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Finishing because we ran out of data.')
        finally:
            bar.finish()
            print('Done.')
            coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    tf.app.run()
