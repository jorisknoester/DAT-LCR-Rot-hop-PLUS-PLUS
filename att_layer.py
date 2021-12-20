# Definitions for attention layers in a neural network.
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


def softmax_with_len(inputs, length, max_len):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param inputs:
    :param length:
    :param max_len:
    :return:
    """
    inputs = tf.cast(inputs, tf.float32)
    inputs = tf.exp(inputs)
    if length is not None:
        length = tf.reshape(length, [-1])
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
        inputs *= mask
    _sum = tf.reduce_sum(inputs, reduction_indices=-1, keep_dims=True) + 1e-9
    return inputs / _sum


def bilinear_attention_layer(inputs, attend, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    Method obtained from Trusca et al. (2020), original docstring below.

    :param inputs: batch * max_len * n_hidden
    :param attend: batch * n_hidden
    :param length:
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id:
    :return:
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.reshape(tf.matmul(inputs, w), [-1, max_len, n_hidden])
    attend = tf.expand_dims(attend, 2)
    tmp = tf.reshape(tf.matmul(tmp, attend), [batch_size, 1, max_len])
    return softmax_with_len(tmp, length, max_len)


def dot_produce_attention_layer(inputs, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    Method obtained from Trusca et al. (2020), original docstring below.

    :param inputs: batch * max_len * n_hidden
    :param length: batch * 1
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id: layer's identical id
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    u = tf.get_variable(
        name='att_u_' + str(layer_id),
        shape=[n_hidden, 1],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + 1))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.reshape(tf.matmul(inputs, u), [batch_size, 1, max_len])
    alpha = softmax_with_len(tmp, length, max_len)
    return alpha
