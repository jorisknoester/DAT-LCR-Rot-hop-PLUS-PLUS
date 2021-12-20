# Methods for loading data.
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

import random

import numpy as np
from sklearn.model_selection import StratifiedKFold

from data_rest_lapt import read_rest_lapt
from config import *


def load_data_and_embeddings(config, load_data):
    """
    Loads data. Method adapted from Trusca et al. (2020), no original docstring provided.

    :param config: configuration
    :param load_data: False for BERT embeddings
    :return:
    """
    flags = config

    if load_data:
        source_count, target_count = [], []
        source_word2idx, target_phrase2idx = {}, {}

        print('reading training data...')
        train_data_source = read_rest_lapt(flags.train_data_source, source_count, source_word2idx, target_count,
                                           target_phrase2idx,
                                           flags.train_path_source)
        train_data_target = read_rest_lapt(flags.train_data_target, source_count, source_word2idx, target_count,
                                           target_phrase2idx,
                                           flags.train_path_target)
        print('reading test data...')
        test_data = read_rest_lapt(flags.test_data, source_count, source_word2idx, target_count, target_phrase2idx,
                                   flags.test_path)

        wt = np.random.normal(0, 0.05, [len(source_word2idx), 300])
        count = 0.0
        with open(flags.pretrain_file, 'r', encoding="utf8") as f:
            for line in f:
                content = line.strip().split()
                if content[0] in source_word2idx:
                    wt[source_word2idx[content[0]]] = np.array(list(map(float, content[1:])))
                    count += 1

        print('finished embedding context vectors...')

        # Print data to txt file.
        out_f = open(flags.embedding_path, "w")
        for i, word in enumerate(source_word2idx):
            out_f.write(word)
            out_f.write(" ")
            out_f.write(' '.join(str(w) for w in wt[i]))
            out_f.write("\n")
        out_f.close()
        print((len(source_word2idx) - count) / len(source_word2idx) * 100)

        return len(train_data_source[0]), len(train_data_target[0]), len(test_data[0]), train_data_source[3], \
               train_data_target[3], test_data[3]

    else:
        # Get statistic properties from the text files for the source, target, and test domain set.
        train_size_source, train_polarity_vector_source = get_stats_from_file(flags.train_path_source)
        train_size_target, train_polarity_vector_target = get_stats_from_file(flags.train_path_target)
        test_size, test_polarity_vector = get_stats_from_file(flags.test_path)

        return train_size_source, train_size_target, test_size, train_polarity_vector_source,\
               train_polarity_vector_target, test_polarity_vector


def get_stats_from_file(path):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param path: file path
    :return:
    """
    polarity_vector = []
    with open(path, "r") as fd:
        lines = fd.read().splitlines()
        size = len(lines) / 3
        for i in range(0, len(lines), 3):
            # Polarity.
            polarity_vector.append(lines[i + 2].strip().split()[0])
    return size, polarity_vector


def load_hyper_data(config, shuffle=True, percentage=0.8):
    """
    Divides the training set in a validation set and a training set. In this case, the test data won't be used for
    hyperparameter finetuning.

    :param config: configuration
    :param shuffle: True if train data need to be split for hyperparameter tuning
    :param percentage: percentage to go to train file
    :return:
    """
    flags = config

    if shuffle:
        """Splits a file in 2 given the `percentage` to go in the large file."""
        random.seed(12345)
        with open(flags.train_path_source, 'r') as fin, \
                open(flags.hyper_train_path_source, 'w') as f_out_big:
            lines = fin.readlines()

            chunked = [lines[i:i + 3] for i in range(0, len(lines), 3)]
            random.shuffle(chunked)
            numlines = int(len(chunked) * percentage)

            # The sample sizes of the three chosen datasets for hyperparameter tuning were all divisble by 120. Having
            # an equal number of batches improves the performance of the CLRH.
            FLAGS.batch_size_src = int(numlines/120)
            if numlines % 20 == 1:
                numlines += 1
            for chunk in chunked[:numlines]:
                for line in chunk:
                    f_out_big.write(line)
        with open(flags.train_path_target, 'r') as fin, \
                open(flags.hyper_train_path_target, 'w') as f_out_big, \
                open(flags.hyper_eval_path_target, 'w') as f_out_small:
            lines = fin.readlines()

            chunked = [lines[i:i + 3] for i in range(0, len(lines), 3)]
            random.shuffle(chunked)
            numlines = int(len(chunked) * percentage)
            # Adjusts the test size to the number of observations in the hyperresults test set.
            FLAGS.batch_size_te = int((len(chunked) - numlines))
            # The sample sizes of the three chosen datasets for hyperparameter tuning were all divisble by 120. Having
            # an equal number of batches improves the performance of the CLRH.
            FLAGS.batch_size_tar = int(numlines/120)
            if numlines % 20 == 1:
                numlines += 1
            for chunk in chunked[:numlines]:
                for line in chunk:
                    f_out_big.write(line)
            for chunk in chunked[numlines:]:
                for line in chunk:
                    f_out_small.write(line)

    # Get statistic properties from txt file.
    train_size_source, train_polarity_vector_source = get_stats_from_file(flags.hyper_train_path_source)
    train_size_target, train_polarity_vector_target = get_stats_from_file(flags.hyper_train_path_target)
    test_size, test_polarity_vector = get_stats_from_file(flags.hyper_eval_path_target)

    return train_size_source, train_size_target, test_size, train_polarity_vector_source, \
            train_polarity_vector_target, test_polarity_vector


def load_cross_validation(config, n_folds, load=True):
    """
    Method adapted from Trusca et al. (2020), no original docstring provided.
    NOTE. Not used in current adaptation.

    :param config: configuration
    :param n_folds: number of cross-validation sets (at least 2)
    :param load: True if data needs to be split into cross-validation sets (defaults to True)
    :return:
    """
    flags = config
    if load:
        words, sent = [], []

        with open(flags.train_path, encoding='cp1252') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                words.append([lines[i], lines[i + 1], lines[i + 2]])
                sent.append(lines[i + 2].strip().split()[0])
            words = np.asarray(words)

            sent = np.asarray(sent)

            i = 1
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=12345)
            for train_idx, val_idx in kf.split(words, sent):
                words_1 = words[train_idx]
                words_2 = words[val_idx]
                with open("data/programGeneratedData/crossValidation/cross_train_" + str(
                        i) + '.txt', 'w') as train, \
                        open("data/programGeneratedData/crossValidation/cross_val_" + str(
                            i) + '.txt', 'w') as val:
                    for row in words_1:
                        train.write(row[0])
                        train.write(row[1])
                        train.write(row[2])
                    for row in words_2:
                        val.write(row[0])
                        val.write(row[1])
                        val.write(row[2])
                i += 1
    # Get statistic properties from txt file.
    train_size, train_polarity_vector = get_stats_from_file(
        "data/programGeneratedData/crossValidation/cross_train_1.txt")
    test_size, test_polarity_vector = [], []
    for i in range(1, n_folds + 1):
        test_size_i, test_polarity_vector_i = get_stats_from_file(
            "data/programGeneratedData/crossValidation/cross_val_" + str(i) + '.txt')
        test_size.append(test_size_i)
        test_polarity_vector.append(test_polarity_vector_i)

    return train_size, test_size, train_polarity_vector, test_polarity_vector
