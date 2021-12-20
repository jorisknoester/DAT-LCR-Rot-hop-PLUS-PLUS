# CLRH method for obtaining cross-domain feature representations. The first component is the LCR-Rot-hop++ model, which
# is expanded with the domain and class discriminator.
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

import os

import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score

from att_layer import bilinear_attention_layer, dot_produce_attention_layer
from config import *
from nn_layer import class_discriminator, domain_discriminator, bi_dynamic_rnn, reduce_mean_with_len
from utils import load_w2v, batch_index, load_inputs_twitter_keep, load_inputs_twitter

sys.path.append(os.getcwd())
tf.set_random_seed(1)


def lcr_rot(input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob, l2, _id='all'):
    """
    Structure of LCR-Rot-hop++ neural network. Method adapted from Trusca et al. (2020), no original docstring provided.

    :param input_fw:
    :param input_bw:
    :param sen_len_fw:
    :param sen_len_bw:
    :param target:
    :param sen_len_tr:
    :param keep_prob:
    :param l2:
    :param _id:
    :return:
    """
    print('I am lcr_rot_hop_plusplus.')
    cell = tf.contrib.rnn.LSTMCell
    # Left Bi-LSTM.   
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob)
    hiddens_l = bi_dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'l' + _id, 'all')

    # Right Bi-LSTM.
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob)
    hiddens_r = bi_dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'r' + _id,
                               'all')

    # Target Bi-LSTM.
    target = tf.nn.dropout(target, keep_prob=keep_prob)
    hiddens_t = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id,
                               'all')
    pool_t = reduce_mean_with_len(hiddens_t, sen_len_tr)

    # Left context attention layer.
    att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                     'l')
    outputs_l_init = tf.matmul(att_l, hiddens_l)
    outputs_l = tf.squeeze(outputs_l_init)

    # Right context attention layer.
    att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                     'r')
    outputs_r_init = tf.matmul(att_r, hiddens_r)
    outputs_r = tf.squeeze(outputs_r_init)

    # Left-aware target attention layer.
    att_t_l = bilinear_attention_layer(hiddens_t, outputs_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                       'tl')
    outputs_t_l_init = tf.matmul(att_t_l, hiddens_t)

    # Right-aware target attention layer.
    att_t_r = bilinear_attention_layer(hiddens_t, outputs_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                       'tr')
    outputs_t_r_init = tf.matmul(att_t_r, hiddens_t)

    # Context and target hierarchical attention layers.
    outputs_init_context = tf.concat([outputs_l_init, outputs_r_init], 1)
    outputs_init_target = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
    att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2,
                                                      FLAGS.random_base, 'fin1')
    att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2,
                                                     FLAGS.random_base, 'fin2')
    outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 0], 2), outputs_l_init))
    outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 1], 2), outputs_r_init))
    outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 0], 2), outputs_t_l_init))
    outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 1], 2), outputs_t_r_init))

    # Add two more hops.
    for i in range(2):
        # Left context attention layer.
        att_l = bilinear_attention_layer(hiddens_l, outputs_t_l, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                         'l' + str(i))
        outputs_l_init = tf.matmul(att_l, hiddens_l)
        outputs_l = tf.squeeze(outputs_l_init)

        # Right context attention layer.
        att_r = bilinear_attention_layer(hiddens_r, outputs_t_r, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                         'r' + str(i))
        outputs_r_init = tf.matmul(att_r, hiddens_r)
        outputs_r = tf.squeeze(outputs_r_init)

        # Left-aware target attention layer.
        att_t_l = bilinear_attention_layer(hiddens_t, outputs_l, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                           FLAGS.random_base, 'tl' + str(i))
        outputs_t_l_init = tf.matmul(att_t_l, hiddens_t)

        # Right-aware target attention layer.
        att_t_r = bilinear_attention_layer(hiddens_t, outputs_r, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                           FLAGS.random_base, 'tr' + str(i))
        outputs_t_r_init = tf.matmul(att_t_r, hiddens_t)

        # Context and target hierarchical attention layers.
        outputs_init_context = tf.concat([outputs_l_init, outputs_r_init], 1)
        outputs_init_target = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
        att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2,
                                                          FLAGS.random_base, 'fin1' + str(i))
        att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2,
                                                         FLAGS.random_base, 'fin2' + str(i))
        outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 0], 2), outputs_l_init))
        outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 1], 2), outputs_r_init))
        outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 0], 2), outputs_t_l_init))
        outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 1], 2), outputs_t_r_init))

    outputs_fin = tf.concat([outputs_l, outputs_r, outputs_t_l, outputs_t_r], 1)
    return outputs_fin, att_l, att_r, att_t_l, att_t_r




def main(train_path_source, train_path_target, test_path, learning_rate_dis=0.01, learning_rate_f=0.01, keep_prob=0.3,
         momentum_dis=0.85, momentum_f=0.85, l2_dis=0.001, l2_f=0.001, balance_lambda=1.0):
    """
    Runs the CLRH neural network. Method adapted from Trusca et al. (2020), no original
    docstring provided.

    :param train_path_source: the training path of the source domain
    :param train_path_target: the training path of the target domain
    :param test_path: the path of the test set
    :param learning_rate_dis: learning rate domain discriminator
    :param learning_rate_f: learning rate feature extractor and class discriminator
    :param keep_prob: keep probability
    :param momentum_dis: momentum factor domain discriminator
    :param momentum_f: momemtum factor feature extractor and class discriminator
    :param l2_dis: l2-regularisation term domain discriminator
    :param l2_f: l2-regularisation term feature extractor and class discriminator
    :param balance_lambda: DANN balance parameter
    :return:
    """
    print_config()
    tf.reset_default_graph()
    with tf.device('/gpu:1'):
        # Obtain embeddings for source, target domain, and test set
        train_word_id_mapping_source, train_w2v_source = load_w2v(FLAGS.train_embedding_source, FLAGS.embedding_dim)
        train_word_embedding_source = tf.constant(train_w2v_source, dtype=np.float32,
                                                  name='train_word_embedding_source')
        train_word_id_mapping_target, train_w2v_target = load_w2v(FLAGS.train_embedding_target, FLAGS.embedding_dim)
        train_word_embedding_target = tf.constant(train_w2v_target, dtype=np.float32,
                                                  name='train_word_embedding_target')
        test_word_id_mapping, test_w2v = load_w2v(FLAGS.test_embedding, FLAGS.embedding_dim)
        test_word_embedding = tf.constant(test_w2v, dtype=np.float32,
                                          name='test_word_embedding')

        keep_prob_all = tf.placeholder(tf.float32)

        with tf.name_scope('inputs'):
            # _src represents source domain, _tar equals the target domain, while _te stands for test set.
            x_src = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])  # input sentence
            y_src = tf.placeholder(tf.float32, [None, FLAGS.n_class])  # sentiment label
            d_src = tf.placeholder(tf.float32, [None, FLAGS.n_domain])  # domain label

            sen_len_src = tf.placeholder(tf.int32, None)
            x_bw_src = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            sen_len_bw_src = tf.placeholder(tf.int32, [None])
            target_words_src = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
            tar_len_src = tf.placeholder(tf.int32, [None])

            x_tar = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])  # input sentence
            d_tar = tf.placeholder(tf.float32, [None, FLAGS.n_domain])  # domain label

            sen_len_tar = tf.placeholder(tf.int32, None)
            x_bw_tar = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            sen_len_bw_tar = tf.placeholder(tf.int32, [None])
            target_words_tar = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
            tar_len_tar = tf.placeholder(tf.int32, [None])

            x_te = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])  # input sentence
            y_te = tf.placeholder(tf.float32, [None, FLAGS.n_class])  # sentiment label

            sen_len_te = tf.placeholder(tf.int32, None)
            x_bw_te = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            sen_len_bw_te = tf.placeholder(tf.int32, [None])
            target_words_te = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
            tar_len_te = tf.placeholder(tf.int32, [None])

        # Get input for the CLRH model
        inputs_fw_source = tf.nn.embedding_lookup(train_word_embedding_source, x_src)
        inputs_bw_source = tf.nn.embedding_lookup(train_word_embedding_source, x_bw_src)
        target_source = tf.nn.embedding_lookup(train_word_embedding_source, target_words_src)
        inputs_fw_target = tf.nn.embedding_lookup(train_word_embedding_target, x_tar)
        inputs_bw_target = tf.nn.embedding_lookup(train_word_embedding_target, x_bw_tar)
        target_target = tf.nn.embedding_lookup(train_word_embedding_target, target_words_tar)
        inputs_fw_test = tf.nn.embedding_lookup(test_word_embedding, x_te)
        inputs_bw_test = tf.nn.embedding_lookup(test_word_embedding, x_bw_te)
        target_test = tf.nn.embedding_lookup(test_word_embedding, target_words_te)

        # Get output of LCR-Rot-hop++ for both the source and target domain.
        with tf.variable_scope("lcr_rot", reuse=tf.AUTO_REUSE) as scope:
            outputs_fin_source, alpha_fw_source, alpha_bw_source, alpha_t_l_source, alpha_t_r_source = lcr_rot(
                inputs_fw_source, inputs_bw_source, sen_len_src, sen_len_bw_src,
                target_source, tar_len_src, keep_prob_all, l2_f, 'source', 'all')
        with tf.variable_scope("lcr_rot", reuse=tf.AUTO_REUSE) as scope:
            outputs_fin_target, alpha_fw_target, alpha_bw_target, alpha_t_l_target, alpha_t_r_target = lcr_rot(
                inputs_fw_target, inputs_bw_target, sen_len_tar, sen_len_bw_tar,
                target_target, tar_len_tar, keep_prob_all, l2_f, 'target', 'all')

        # Pass both domains through the domain discriminator and compute its combined loss
        with tf.variable_scope("dis",
                               reuse=tf.AUTO_REUSE) as scope:
            prob_domain_source, w0, w1 = domain_discriminator(outputs_fin_source, 8 * FLAGS.n_hidden, keep_prob_all,
                                                              l2_dis, FLAGS.n_domain)
            prob_domain_target, w0, w1 = domain_discriminator(outputs_fin_target, 8 * FLAGS.n_hidden, keep_prob_all,
                                                              l2_dis, FLAGS.n_domain)

        # Only add the l2-regularisation term once for the weights.
        loss_domain_source = loss_func_domain_discr(d_src, prob_domain_source, w0, w1, True)
        loss_domain_target = loss_func_domain_discr(d_tar, prob_domain_target, w0, w1, False)
        acc_num_domain_source, acc_prob_domain_source = acc_func(d_src, prob_domain_source)
        acc_num_domain_target, acc_prob_domain_target = acc_func(d_tar, prob_domain_target)
        loss_domain = loss_domain_target + loss_domain_source
        acc_num_domain = acc_num_domain_source + acc_num_domain_target

        # Pass only the source domain through the class discriminator and calculate its loss
        with tf.variable_scope("class", reuse=tf.AUTO_REUSE) as scope:
            prob_class = class_discriminator(outputs_fin_source, 8 * FLAGS.n_hidden, keep_prob_all, l2_f, FLAGS.n_class)
        loss_class = loss_func_class_discr(y_src, prob_class)
        acc_num_class, acc_prob_class = acc_func(y_src, prob_class)

        # Define total loss of CLRH++ model
        loss_f = loss_class - balance_lambda * loss_domain

        # Set step variable and obtain the different variable lists to be used for the training
        global_step = tf.Variable(0, name='tr_global_step', trainable=False)
        var_list_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')
        var_list_C = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='class')
        trainable = tf.trainable_variables()
        var_list_f = trainable[:30] + var_list_C

        # opti_min_domain maximises the loss, whereas opti_feature minimises the loss in order to achieve Domain
        # Adversarial Training.
        opti_min_domain = tf.train.MomentumOptimizer(learning_rate=learning_rate_dis, momentum=momentum_dis) \
            .minimize(-loss_f, var_list=var_list_D, global_step=global_step)
        opti_feature = tf.train.MomentumOptimizer(learning_rate=learning_rate_f, momentum=momentum_f) \
            .minimize(loss_f, var_list=var_list_f, global_step=global_step)

        # Feed the test set through both the LCR-Rot-hop++ and the class discriminator to predict its sentiments.
        with tf.variable_scope("lcr_rot", reuse=True) as scope:
            outputs_fin_test, alpha_fw_test, alpha_bw_test, alpha_t_l_test, alpha_t_r_test = lcr_rot(
                inputs_fw_test, inputs_bw_test, sen_len_te, sen_len_bw_te,
                target_test, tar_len_te, keep_prob_all, l2_f, 'target', 'all')
        with tf.variable_scope("class", reuse=True) as scope:
            prob_class_test = class_discriminator(outputs_fin_test, 8 * FLAGS.n_hidden, keep_prob_all, l2_f, FLAGS.n_class)
        loss_class_test = loss_func_class_discr(y_te, prob_class_test)
        acc_num_class_test, acc_prob_class_test = acc_func(y_te, prob_class_test)
        # Define the predicted label and real label
        pred_y = tf.argmax(prob_class_test, 1)
        true_y = tf.argmax(y_te, 1)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Initialize weights and biases.
        sess.run(tf.global_variables_initializer())

        if FLAGS.is_r == '1':
            is_r = True
        else:
            is_r = False

        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write("Train data source. ")
        tr_x_src, tr_sen_len_src, tr_x_bw_src, tr_sen_len_bw_src, tr_y_src, tr_target_word_src, tr_tar_len_src, \
        _, _, _, y_onehot_mapping_src = load_inputs_twitter(
            train_path_source,
            train_word_id_mapping_source,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )
        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write("Train data target. ")
        tr_x_tar, tr_sen_len_tar, tr_x_bw_tar, tr_sen_len_bw_tar, tr_y_tar, tr_target_word_tar, tr_tar_len_tar, \
        _, _, _, y_onehot_mapping_tar = load_inputs_twitter(
            train_path_target,
            train_word_id_mapping_target,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )

        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write("Test data. ")
        te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, _, _, _, \
        y_onehot_mapping_te = load_inputs_twitter(
            test_path,
            test_word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )

        def get_batch_data(x_f_src, x_f_tar, x_f_te, sen_len_f_src, sen_len_f_tar, sen_len_f_te, x_b_src, x_b_tar,
                           x_b_te, sen_len_b_src, sen_len_b_tar, sen_len_b_te, yi_src, yi_tar, yi_te, batch_target_src,
                           batch_target_tar, batch_target_te, batch_tl_src, batch_tl_tar, batch_tl_te, batch_size_src,
                           batch_size_tar, batch_size_te, keep_pr, domain_src, domain_tar, run_test, is_shuffle=True):
            """
            Obtains the batches for each iteration of the CLRH method. All parameters with name_scope('inputs') should
            be defined here.

            :param x_f_src:
            :param x_f_tar:
            :param x_f_te:
            :param sen_len_f_src:
            :param sen_len_f_tar:
            :param sen_len_f_te:
            :param x_b_src:
            :param x_b_tar:
            :param x_b_te:
            :param sen_len_b_src:
            :param sen_len_b_tar:
            :param sen_len_b_te:
            :param yi_src:
            :param yi_tar:
            :param yi_te:
            :param batch_target_src:
            :param batch_target_tar:
            :param batch_tl_src:
            :param batch_target_te:
            :param batch_tl_tar:
            :param batch_tl_te:
            :param batch_size_src:
            :param batch_size_tar:
            :param batch_size_te:
            :param keep_pr:
            :param domain_src:
            :param domain_tar:
            :param run_test: True if testing
            :param is_shuffle: True if the trainings sets must be shuffled
            :return:
            """
            for index_src, index_tar, index_te in batch_index(len(yi_src), len(yi_tar), len(yi_te), batch_size_src,
                                                              batch_size_tar, batch_size_te, is_shuffle, run_test):
                feed_dict = {
                    x_src: x_f_src[index_src],
                    x_bw_src: x_b_src[index_src],
                    y_src: yi_src[index_src],
                    sen_len_src: sen_len_f_src[index_src],
                    sen_len_bw_src: sen_len_b_src[index_src],
                    target_words_src: batch_target_src[index_src],
                    tar_len_src: batch_tl_src[index_src],
                    d_src: domain_src[index_src],
                    x_tar: x_f_tar[index_tar],
                    x_bw_tar: x_b_tar[index_tar],
                    sen_len_tar: sen_len_f_tar[index_tar],
                    sen_len_bw_tar: sen_len_b_tar[index_tar],
                    target_words_tar: batch_target_tar[index_tar],
                    tar_len_tar: batch_tl_tar[index_tar],
                    d_tar: domain_tar[index_tar],
                    x_te: x_f_te[index_te],
                    x_bw_te: x_b_te[index_te],
                    sen_len_te: sen_len_f_te[index_te],
                    sen_len_bw_te: sen_len_b_te[index_te],
                    target_words_te: batch_target_te[index_te],
                    tar_len_te: batch_tl_te[index_te],
                    y_te: yi_te[index_te],
                    keep_prob_all: keep_pr,
                }
                # For testing return the test size, otherwise return both source and target batch size for accuracy
                # computation.
                if run_test:
                    yield feed_dict, len(index_te)
                else:
                    yield feed_dict, len(index_src), len(index_tar)

        max_acc = 0.
        for i in range(FLAGS.n_iter):
            # Initialise performance variables
            train_count = 0
            train_count_tar = 0
            domain_trainacc = 0
            class_trainacc = 0
            # Specify the domain vectors for both the source and target domain data.
            src_domain = np.zeros((len(tr_y_src), 2))
            src_domain[:, 0] = 1
            tar_domain = np.zeros((len(tr_y_tar), 2))
            tar_domain[:, 1] = 1

            # Train model.
            for train, numtrain, train_count_t in get_batch_data(tr_x_src, tr_x_tar, te_x, tr_sen_len_src,
                                                                 tr_sen_len_tar, te_sen_len, tr_x_bw_src,
                                                                 tr_x_bw_tar, te_x_bw, tr_sen_len_bw_src,
                                                                 tr_sen_len_bw_tar, te_sen_len_bw,
                                                                 tr_y_src, tr_y_tar, te_y,
                                                                 tr_target_word_src,
                                                                 tr_target_word_tar, te_target_word,
                                                                 tr_tar_len_src, tr_tar_len_tar, te_tar_len,
                                                                 FLAGS.batch_size_src,
                                                                 FLAGS.batch_size_tar, FLAGS.batch_size_te, keep_prob,
                                                                 src_domain, tar_domain, False):
                train_count += numtrain
                train_count_tar += train_count_t
                _, _, step, _domain_trainacc, _class_trainacc = sess.run(
                    [opti_min_domain, opti_feature, global_step, acc_num_domain, acc_num_class],
                    feed_dict=train)
                domain_trainacc += _domain_trainacc
                class_trainacc += _class_trainacc

            acc, cost, cnt = 0., 0., 0
            fw, bw, tl, tr, ty, py = [], [], [], [], [], []
            p = []
            # Test model.
            for test, num in get_batch_data(tr_x_src, tr_x_tar, te_x, tr_sen_len_src,
                                            tr_sen_len_tar, te_sen_len, tr_x_bw_src,
                                            tr_x_bw_tar, te_x_bw, tr_sen_len_bw_src,
                                            tr_sen_len_bw_tar, te_sen_len_bw,
                                            tr_y_src, tr_y_tar, te_y,
                                            tr_target_word_src,
                                            tr_target_word_tar, te_target_word,
                                            tr_tar_len_src, tr_tar_len_tar, te_tar_len, FLAGS.batch_size_te,
                                            FLAGS.batch_size_te, FLAGS.batch_size_te, 1.0, src_domain,
                                            tar_domain, True):
                if FLAGS.method == 'TD-ATT' or FLAGS.method == 'IAN':
                    _loss, _acc, _fw, _bw, _tl, _tr, _ty, _py, _p = sess.run(
                        [loss_class_test, acc_num_class_test, alpha_fw_test, alpha_bw_test, alpha_t_l_test,
                         alpha_t_r_test, true_y, pred_y, prob_class_test], feed_dict=test)
                    fw += list(_fw)
                    bw += list(_bw)
                    tl += list(_tl)
                    tr += list(_tr)
                else:
                    _loss, _acc, _ty, _py, _p, _fw, _bw, _tl, _tr = sess.run(
                        [loss_class_test, acc_num_class_test, true_y, pred_y, prob_class_test, alpha_fw_test,
                         alpha_bw_test,
                         alpha_t_l_test, alpha_t_r_test], feed_dict=test)
                ty = np.asarray(_ty)
                py = np.asarray(_py)
                p = np.asarray(_p)
                fw = np.asarray(_fw)
                bw = np.asarray(_bw)
                tl = np.asarray(_tl)
                tr = np.asarray(_tr)
                acc += _acc
                cost += _loss * num
                cnt += num
            print('Total samples={}, correct predictions={}'.format(cnt, acc))
            # Compute performance statistics
            class_trainacc = class_trainacc / train_count
            domain_trainacc = domain_trainacc / (train_count + train_count_tar)
            acc = acc / cnt
            cost = cost / cnt
            print(
                'Iter {}: mini-batch loss={:.6f}, train domain acc={:.6f}, class acc={:.6f}, test acc={:.6f}'.format(i,
                                                                                                                     cost,
                                                                                                                     domain_trainacc,
                                                                                                                     class_trainacc,
                                                                                                                     acc))
            if acc > max_acc:
                max_acc = acc
        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write(
                    "---\nLCR-Rot-Hop++. Train accuracy: {:.6f}, Test accuracy: {:.6f}\n".format(
                        class_trainacc, acc))
                results.write("Maximum. Test accuracy: {:.6f}\n---\n".format(max_acc))

        precision = precision_score(ty, py, average=None)
        recall = recall_score(ty, py, average=None)
        f1 = f1_score(ty, py, average=None)
        print('\nP:', precision, 'avg=', sum(precision) / FLAGS.n_class)
        print('R:', recall, 'avg=', sum(recall) / FLAGS.n_class)
        print('F1:', f1, 'avg=', str(sum(f1) / FLAGS.n_class) + '\n')

        with open(FLAGS.prob_file + '.txt', 'w') as fp:
            for item in p:
                fp.write(' '.join([str(it) for it in item]) + '\n')
        with open(FLAGS.prob_file + '_fw.txt', 'w') as fp:
            for y1, y2, ws in zip(ty, py, fw):
                fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        with open(FLAGS.prob_file + '_bw.txt', 'w') as fp:
            for y1, y2, ws in zip(ty, py, bw):
                fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        with open(FLAGS.prob_file + '_tl.txt', 'w') as fp:
            for y1, y2, ws in zip(ty, py, tl):
                fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        with open(FLAGS.prob_file + '_tr.txt', 'w') as fp:
            for y1, y2, ws in zip(ty, py, tr):
                fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')

        print('Optimization Finished! Test accuracy={}\n'.format(max_acc))

        # Save model if savable.
        if FLAGS.savable == 1:
            save_dir = "model/" + FLAGS.source_domain + "_" + FLAGS.target_domain + "/"
            saver = saver_func(save_dir)
            saver.save(sess, save_dir)

        # Record accuracy by polarity.
        FLAGS.pos = y_onehot_mapping_te['1']
        FLAGS.neu = y_onehot_mapping_te['0']
        FLAGS.neg = y_onehot_mapping_te['-1']
        pos_count = 0
        neg_count = 0
        neu_count = 0
        pos_correct = 0
        neg_correct = 0
        neu_correct = 0
        for i in range(0, len(ty)):
            if ty[i] == FLAGS.pos:
                # Positive sentiment.
                pos_count += 1
                if py[i] == FLAGS.pos:
                    pos_correct += 1
            elif ty[i] == FLAGS.neu:
                # Neutral sentiment.
                neu_count += 1
                if py[i] == FLAGS.neu:
                    neu_correct += 1
            else:
                # Negative sentiment.
                neg_count += 1
                if py[i] == FLAGS.neg:
                    neg_correct += 1
        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write("Test results.\n")
                results.write(
                    "Positive. Correct: {}, Incorrect: {}, Total: {}\n".format(pos_correct, pos_count - pos_correct,
                                                                               pos_count))
                results.write(
                    "Neutral. Correct: {}, Incorrect: {}, Total: {}\n".format(neu_correct, neu_count - neu_correct,
                                                                              neu_count))
                results.write(
                    "Negative. Correct: {}, Incorrect: {}, Total: {}\n---\n".format(neg_correct,
                                                                                    neg_count - neg_correct,
                                                                                    neg_count))

        return acc, np.where(np.subtract(py, ty) == 0, 0,
                             1), fw.tolist(), bw.tolist(), tl.tolist(), tr.tolist()

    if __name__ == '__main__':
        tf.app.run()
