#!/usr/bin/env python
# encoding: utf-8

# Configuration file.
#
# https://github.com/jorisknoester/DAT-LCR-Rot-hop++
#
# Adapted from Trusca, Wassenberg, Frasincar and Dekker (2020).
# https://github.com/mtrusca/HAABSA_PLUS_PLUS
#
# Truşcǎ M.M., Wassenberg D., Frasincar F., Dekker R. (2020) A Hybrid Approach for Aspect-Based Sentiment Analysis Using
# Deep Contextual Word Embeddings and Hierarchical Attention. In: Bielikova M., Mikkonen T., Pautasso C. (eds) Web
# Engineering. ICWE 2020. Lecture Notes in Computer Science, vol 12128. Springer, Cham.
# https://doi.org/10.1007/978-3-030-50578-3_25

import sys
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = tf.app.flags.FLAGS
neutral_sentiment = True

# Implementation settings.
tf.app.flags.DEFINE_string("embedding_type", "BERT", "type of embedding used (BERT or glove)")
tf.app.flags.DEFINE_integer('embedding_dim', 768, 'dimension of word embed H')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
tf.app.flags.DEFINE_integer('n_domain', 2, 'number of distinct domains')
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 20, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('random_base', 0.01, 'initial random base')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('pos', 0, 'mapping for positive sentiment')
tf.app.flags.DEFINE_integer('neg', 1, 'mapping for negative sentiment')
tf.app.flags.DEFINE_integer('source_domain_mapping', 0, 'mapping for source domain')
tf.app.flags.DEFINE_integer('target_domain_mapping', 1, 'mapping for target domain')
tf.app.flags.DEFINE_string('t1', 'last', 'type of hidden output')
tf.app.flags.DEFINE_string('t2', 'last', 'type of hidden output')
tf.app.flags.DEFINE_integer('hidden_layers', 1, 'number of layers in class and domain discriminator')
tf.app.flags.DEFINE_integer('n_layer', 3, 'number of stacked rnn')
tf.app.flags.DEFINE_string('is_r', '1', 'prob')
tf.app.flags.DEFINE_integer('max_target_len', 19, 'max target length')
tf.app.flags.DEFINE_integer('splits', 9, 'number of data splits for test')
tf.app.flags.DEFINE_integer('neutral_sentiment', 0, 'Neutral sentiment in/excluded')
tf.app.flags.DEFINE_integer('neutral_to_negative', 0, 'Neutral sentiment as part of negative class')
tf.app.flags.DEFINE_integer('l2_regularization', 1, 'L2-regularization on or off')
if neutral_sentiment:
    FLAGS.neutral_sentiment = 1
    FLAGS.n_class = 3
    FLAGS.neg = 2
    tf.app.flags.DEFINE_integer('neu', 1, 'mapping for neutral sentiment')

# The domain settings
tf.app.flags.DEFINE_string("source_domain", "book", "source domain (training set)")
tf.app.flags.DEFINE_string("target_domain", "restaurant", "target domain (test set)")
tf.app.flags.DEFINE_integer("source_year", 2019, "year data set")
tf.app.flags.DEFINE_integer("target_year", 2014, "year data set")
tf.app.flags.DEFINE_integer('batch_size_src', 24, 'number of example per batch_src')
tf.app.flags.DEFINE_integer('batch_size_tar', 15, 'number of example per batch_tar')
tf.app.flags.DEFINE_integer('batch_size_te', 701, 'number of example in test set')
tf.app.flags.DEFINE_integer('n_iter', 20, 'number of train iter')

# Hyperparameter settings
tf.app.flags.DEFINE_float('balance_lambda', 1.0, 'lambda')
tf.app.flags.DEFINE_float('learning_rate_dis', 0.01, 'learning rate domain discriminator')
tf.app.flags.DEFINE_float('learning_rate_f', 0.01, 'learning rate feature extractor and class discrimiator')
tf.app.flags.DEFINE_float('momentum_dis', 0.85, 'momentum rate domain discriminator')
tf.app.flags.DEFINE_float('momentum_f', 0.85, 'momentum rate feature extractor and class discriminator')
tf.app.flags.DEFINE_float('l2_dis', 0.001, 'l2 regularization domain discriminator')
tf.app.flags.DEFINE_float('l2_f', 0.001, 'l2 regularization feature extractor and class discriminator')
tf.app.flags.DEFINE_float('keep_prob', 0.3, 'keep probability')

# Traindata, testdata and single-domain embeddings.
tf.app.flags.DEFINE_string("train_path_source", "data/programGeneratedData/BERT/" + str(FLAGS.source_domain) + "/" + str(
    FLAGS.embedding_dim) + FLAGS.source_domain + "_train_" + str(FLAGS.source_year) + ".txt", "train source data path")
tf.app.flags.DEFINE_string("train_path_target", "data/programGeneratedData/BERT/" + str(FLAGS.target_domain) + "/" + str(
    FLAGS.embedding_dim) + FLAGS.target_domain + "_train_" + str(FLAGS.target_year) + ".txt", "train target data path")
tf.app.flags.DEFINE_string("test_path", "data/programGeneratedData/BERT/" + str(FLAGS.target_domain) + "/" + str(
    FLAGS.embedding_dim) + str(FLAGS.target_domain) + "_test_" + str(FLAGS.target_year) + ".txt", "formatted test data path")
tf.app.flags.DEFINE_string("embedding_path",
                           "data/programGeneratedData/" + str(FLAGS.embedding_dim) + "embedding" + str(
                               FLAGS.source_year) + ".txt", "pre-trained embedding vectors file path")

# Cross-domain source and target embedding
tf.app.flags.DEFINE_string("train_embedding_source",
                           "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.source_domain + "_" + str(
                               FLAGS.source_year) + "_" + str(FLAGS.embedding_dim) + ".txt",
                           "source domain pre-trained BERT embeddings")
tf.app.flags.DEFINE_string("train_embedding_target",
                           "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
                               FLAGS.target_year) + "_" + str(FLAGS.embedding_dim) + ".txt",
                           "target domain pre-trained BERT embeddings")
tf.app.flags.DEFINE_string("test_embedding",
                           "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
                               FLAGS.target_year) + "_" + str(FLAGS.embedding_dim) + ".txt",
                           "test domain pre-trained BERT embeddings")

# Hyperparameter tuning train and test data.
tf.app.flags.DEFINE_string("hyper_train_path_source",
                           "data/programGeneratedData/" + str(FLAGS.embedding_dim) + 'hypertraindatasource' + "_" + str(
                               FLAGS.source_domain) + ".txt", "hyper train data path")
tf.app.flags.DEFINE_string("hyper_train_path_target",
                           "data/programGeneratedData/" + str(FLAGS.embedding_dim) + 'hypertraindatatarget' + "_" + str(
                               FLAGS.target_domain) + ".txt", "hyper train data path")
tf.app.flags.DEFINE_string("hyper_eval_path_target",
                           "data/programGeneratedData/" + str(FLAGS.embedding_dim) + 'hyperevaldatatarget' + "_" + str(
                               FLAGS.source_domain) + "_" + str(FLAGS.target_domain) + ".txt", "hyper eval data path target")

tf.app.flags.DEFINE_string("hyper_svm_train_path",
                           "data/programGeneratedData/" + str(FLAGS.embedding_dim) + 'hypertrainsvmdata' + str(
                               FLAGS.source_year) + ".txt", "hyper train svm data path")
tf.app.flags.DEFINE_string("hyper_svm_eval_path",
                           "data/programGeneratedData/" + str(FLAGS.embedding_dim) + 'hyperevalsvmdata' + str(
                               FLAGS.source_year) + ".txt", "hyper eval svm data path")

# External data sources.
tf.app.flags.DEFINE_string("pretrain_file", "data/externalData/glove.42B." + str(FLAGS.embedding_dim) + "d.txt",
                           "pre-trained glove vectors file path")
tf.app.flags.DEFINE_string("train_data_source",
                           "data/externalData/" + FLAGS.source_domain + "_train_" + str(FLAGS.source_year) + ".xml",
                           "train data path")
tf.app.flags.DEFINE_string("train_data_target",
                           "data/externalData/" + FLAGS.target_domain + "_train_" + str(FLAGS.target_year) + ".xml",
                           "train data path")
tf.app.flags.DEFINE_string("test_data",
                           "data/externalData/" + FLAGS.target_domain + "_test_" + str(FLAGS.target_year) + ".xml",
                           "test data path")

tf.app.flags.DEFINE_string('method', 'AE', 'model type: AE, AT or AEAT')
tf.app.flags.DEFINE_string('prob_file', 'prob_' + str(FLAGS.source_domain) + "_" + str(FLAGS.target_domain), 'prob')

# Test results.
tf.app.flags.DEFINE_string("results_file", "data/programGeneratedData/" + str(
    FLAGS.embedding_dim) + "results_" + FLAGS.source_domain + "_" + FLAGS.target_domain + "_" + str(
    FLAGS.target_year) + "_" + str(FLAGS.balance_lambda) + ".txt", "results for test")
tf.app.flags.DEFINE_integer("writable", 1, "one if writable")
tf.app.flags.DEFINE_integer("savable", 1, "one if savable")




def print_config():
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.
    """
    FLAGS(sys.argv)
    print('\nParameters:')
    for k, v in sorted(tf.app.flags.FLAGS.flag_values_dict().items()):
        print('{}={}'.format(k, v))
    print()


def loss_func_class_discr(y, prob, weights):
    """
    Calculates the loss of the class discriminator

    :param y: label
    :param prob: predicted probability
    :return: class loss
    """
    loss = - tf.reduce_mean(y * tf.log(prob+0.0001))
    if FLAGS.l2_regularization == 1:
        for w in weights:
                loss += FLAGS.l2_f * tf.nn.l2_loss(w)
    return loss


def loss_func_domain_discr(d, prob, weights, flag):
    """
    Calculates the loss of the domain discriminator

    :param d: domain label
    :param prob: predicted probability
    :param weights: weights of second layer of domain discriminator
    :param flag: boolean that indicates whether or not the regularisation term of the weights should be added
    :return: domain loss
    """
    loss = - tf.reduce_mean(d * tf.log(prob+0.0001))
    if FLAGS.l2_regularization == 1:
        if flag:
            for w in weights:
                loss += FLAGS.l2_dis * tf.nn.l2_loss(w)
    return loss


def acc_func(y, prob):
    """
    Computes accuracy of class and domain discriminator

    :param y: label or domain
    :param prob: predicted probability
    :return: accuracy count and accuracy probability
    """
    correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    acc_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
    acc_prob = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc_num, acc_prob


def summary_func(loss, acc, test_loss, test_acc, _dir, title, sess):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.
    NOTE. Not used in current adaptation.

    :param loss:
    :param acc:
    :param test_loss:
    :param test_acc:
    :param _dir:
    :param title:
    :param sess:
    :return:
    """
    summary_loss = tf.summary.scalar('loss' + title, loss)
    summary_acc = tf.summary.scalar('acc' + title, acc)
    test_summary_loss = tf.summary.scalar('loss' + title, test_loss)
    test_summary_acc = tf.summary.scalar('acc' + title, test_acc)
    train_summary_op = tf.summary.merge([summary_loss, summary_acc])
    validate_summary_op = tf.summary.merge([summary_loss, summary_acc])
    test_summary_op = tf.summary.merge([test_summary_loss, test_summary_acc])
    train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
    test_summary_writer = tf.summary.FileWriter(_dir + '/test')
    validate_summary_writer = tf.summary.FileWriter(_dir + '/validate')
    return train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
           validate_summary_writer


def saver_func(_dir):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param _dir:
    :return:
    """
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000)
    import os
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return saver
