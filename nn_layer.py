# Definitions for layers in a neural network.
#
# https://github.com/jorisknoester/DAT-LCR-Rot-hop-PLUS-PLUS
#
# Adapted from Trusca, Wassenberg, Frasincar and Dekker (2020).
# https://github.com/mtrusca/HAABSA_PLUS_PLUS
#
# Truşcǎ M.M., Wassenberg D., Frasincar F., Dekker R. (2020) A Hybrid Approach for Aspect-Based Sentiment Analysis Using
# Deep Contextual Word Embeddings and Hierarchical Attention. In: Bielikova M., Mikkonen T., Pautasso C. (eds) Web
# Engineering. ICWE 2020. Lecture Notes in Computer Science, vol 12128. Springer, Cham.
# https://doi.org/10.1007/978-3-030-50578-3_25

import numpy as np
import tensorflow as tf


def dynamic_rnn(cell, inputs, n_hidden, length, max_len, scope_name, out_type='last'):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.
    NOTE. Not used in current adaptation.

    :param cell:
    :param inputs:
    :param n_hidden:
    :param length:
    :param max_len:
    :param scope_name:
    :param out_type:
    :return:
    """
    outputs, state = tf.nn.dynamic_rnn(
        cell(n_hidden),
        inputs=inputs,
        dtype=tf.float32,
        scope=scope_name
    )  # outputs -> batch_size * max_len * n_hidden
    batch_size = tf.shape(outputs)[0]
    if out_type == 'last':
        index = tf.range(0, batch_size) * max_len + (length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)  # batch_size * n_hidden
    elif out_type == 'all_avg':
        outputs = reduce_mean_with_len(outputs, length)
    mask = tf.reverse(tf.cast(tf.sequence_mask(length, max_len), tf.float32), [1])
    mask_tiled = tf.tile(mask, [1, n_hidden])
    mask_3d = tf.reshape(mask_tiled, tf.shape(outputs))
    return tf.multiply(outputs, mask_3d)


def bi_dynamic_rnn(cell, inputs, n_hidden, length, max_len, scope_name, out_type='last'):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param cell:
    :param inputs:
    :param n_hidden:
    :param length:
    :param max_len:
    :param scope_name:
    :param out_type:
    :return:
    """
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell(n_hidden),
        cell_bw=cell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope_name
    )
    if out_type == 'last':
        outputs_fw, outputs_bw = outputs
        outputs_bw = tf.reverse_sequence(outputs_bw, tf.cast(length, tf.int64), seq_dim=1)
        outputs = tf.concat([outputs_fw, outputs_bw], 2)
    else:
        outputs = tf.concat(outputs, 2)  # batch_size * max_len * 2n_hidden
    batch_size = tf.shape(outputs)[0]
    if out_type == 'last':
        index = tf.range(0, batch_size) * max_len + (length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, 2 * n_hidden]), index)  # batch_size * 2n_hidden
    elif out_type == 'all_avg':
        outputs = reduce_mean_with_len(outputs, length)  # batch_size * 2n_hidden
    return outputs


def reduce_mean_with_len(inputs, length):
    """
    Method obtained from Trusca et al. (2020), original docstring below.

    :param inputs: 3-D tensor
    :param length: the length of dim [1]
    :return: 2-D tensor
    """
    length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
    inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length
    return inputs


def class_discriminator(inputs, n_hidden, keep_prob, l2, n_class, scope_name='1'):
    """
    The structure of the class discriminator

    :param inputs: the input vector
    :param n_hidden: the number of hidden neurons
    :param keep_prob: keep probability
    :param l2: l2-regularisation term
    :param n_class: number of possible sentiments (3)
    :param scope_name: string to add to weight name
    :return:
    """
    w0 = tf.get_variable(
        name='softmax_w0' + scope_name,
        shape=[n_hidden, n_hidden/4],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_class))),
        regularizer=tf.contrib.layers.l2_regularizer(l2)
    )
    b0 = tf.get_variable(
        name='softmax_b0' + scope_name,
        shape=[n_hidden/4],
        initializer=tf.zeros_initializer(),
    )
    w1 = tf.get_variable(
        name='softmax_w1' + scope_name,
        shape=[n_hidden/4, n_class],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_class))),
        regularizer=tf.contrib.layers.l2_regularizer(l2)
    )
    b1 = tf.get_variable(
        name='softmax_b1' + scope_name,
        shape=[n_class],
        initializer=tf.zeros_initializer(),
    )
    with tf.name_scope('softmax'):
        hidden_output_0 = tf.nn.dropout(inputs, keep_prob=keep_prob)
        hidden_output_0 = tf.matmul(hidden_output_0, w0) + b0
        output = tf.nn.dropout(hidden_output_0, keep_prob=keep_prob)
        output = tf.matmul(output, w1) + b1
        predict = tf.nn.softmax(output)
    return predict


def domain_discriminator(inputs, n_hidden, keep_prob, l2, n_domain, scope_name='1'):
    """
    Structure of the domain discriminator

    :param inputs: the input vector of the domain discriminator
    :param n_hidden: the number of hidden neurons
    :param keep_prob: keep probability
    :param l2: l2-regularisation term
    :param n_domain: number of possible sentiments (2)
    :param scope_name: string to add to weight name
    """
    w0 = tf.get_variable(
        name='dis_w0' + scope_name,
        shape=[8 * n_hidden, 2 * n_hidden],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (8 * n_hidden + n_domain))),
    )
    b0 = tf.get_variable(
        name='dis_b0' + scope_name,
        shape=[2 * n_hidden],
        initializer=tf.zeros_initializer(),
    )
    w1 = tf.get_variable(
        name='dis_w1' + scope_name,
        shape=[2 * FLAGS.n_hidden, FLAGS.n_domain],
        initializer=tf.random_normal_initializer(mean=0.,
                                                 stddev=np.sqrt(2. / (8 * FLAGS.n_hidden + FLAGS.n_domain))),
    )
    b1 = tf.get_variable(
        name='dis_b1' + scope_name,
        shape=[FLAGS.n_domain],
        initializer=tf.zeros_initializer(),
    )
    print('I am the Discriminator')
    hidden_output_0 = tf.nn.dropout(inputs, keep_prob=keep_prob)
    hidden_output_0 = tf.matmul(hidden_output_0, w0) + b0
    outputs = tf.nn.dropout(hidden_output_0, keep_prob=keep_prob)
    outputs = tf.matmul(outputs, w1) + b1
    prob = tf.nn.softmax(outputs)

    return prob, w0, w1

