# Data reader for restaurant and laptop data.
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
import re
import xml.etree.ElementTree as ET
from collections import Counter

import nltk


def window(iterable, size):  # Stack overflow solution for sliding window.
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param iterable:
    :param size:
    :return:
    """
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win


def _get_data_tuple(sptoks, asp_termIn, label):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param sptoks:
    :param asp_termIn:
    :param label:
    :return:
    """
    # Find the ids of aspect term.
    aspect_is = []
    asp_term = ' '.join(sp for sp in asp_termIn).lower()
    for _i, group in enumerate(window(sptoks, len(asp_termIn))):
        if asp_term == ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i, _i + len(asp_termIn)))
            break
        elif asp_term in ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i, _i + len(asp_termIn)))
            break

    pos_info = []
    for _i, sptok in enumerate(sptoks):
        pos_info.append(min([abs(_i - i) for i in aspect_is]))

    lab = None
    if label == 'negative':
        lab = -1
    elif label == 'neutral':
        lab = 0
    elif label == "positive":
        lab = 1
    else:
        raise ValueError("Unknown label: %s" % lab)

    return pos_info, lab


def read_rest_lapt(in_file, source_count, source_word2idx, target_count, target_phrase2idx, out_file):
    """
    Reads data for the 2014 restaurant and laptop dataset. Method adapted from Trusca et al. (2020).

    :param in_file: xml data file location
    :param source_count: list that contains list [<pad>, 0] at the first position [empty input] and all the unique words with number of occurences as tuples [empty input]
    :param source_word2idx: dictionary with unique words and unique index [empty input]
    :param target_count: list that contains list [<pad>, 0] at the first position [empty input] and all the unique words with number of occurences as tuples [empty input]
    :param target_phrase2idx: dictionary with unique words and unique index [empty input]
    :param out_file: file path for output
    :return: tuple specified in function
    """
    # Returns:
    # source_data: list with lists which contain the sentences corresponding to the aspects saved by word indices
    # target_data: list which contains the indices of the target phrases: THIS DOES NOT CORRESPOND TO THE INDICES OF source_data
    # source_loc_data: list with lists which contains the distance from the aspect for every word in the sentence corresponding to the aspect
    # target_label: contains the polarity of the aspect (0=negative, 1=neutral, 2=positive)
    # max_sen_len: maximum sentence length
    # max_target_len: maximum target length

    if not os.path.isfile(in_file):
        raise ("[!] Data %s not found" % in_file)

    # Parse xml file to tree.
    tree = ET.parse(in_file)
    root = tree.getroot()

    out_f = open(out_file, "w")

    # Save all words in source_words (includes duplicates).
    # Save all aspects in target_words (includes duplicates).
    # Sinds max sentence length and max targets length.
    source_words, target_words, max_sent_len, max_target_len = [], [], 0, 0
    target_phrases = []

    count_confl = 0
    for sentence in root.iter('sentence'):
        sent = sentence.find('text').text
        sentence_new = re.sub(' +', ' ', sent)
        sptoks = nltk.word_tokenize(sentence_new)
        for sp in sptoks:
            source_words.extend([''.join(sp).lower()])
        if len(sptoks) > max_sent_len:
            max_sent_len = len(sptoks)
        for aspectTerms in sentence.iter('aspectTerms'):
            for aspectTerm in aspectTerms.findall('aspectTerm'):
                if aspectTerm.get("polarity") == "conflict":
                    count_confl += 1
                    continue
                asp = aspectTerm.get('term')
                if asp != 'NULL':
                    asp_new = re.sub(' +', ' ', asp)
                    t_sptoks = nltk.word_tokenize(asp_new)
                    for sp in t_sptoks:
                        target_words.extend([''.join(sp).lower()])
                    target_phrases.append(' '.join(sp for sp in t_sptoks).lower())
                    if len(t_sptoks) > max_target_len:
                        max_target_len = len(t_sptoks)
    if len(source_count) == 0:
        source_count.append(['<pad>', 0])
    source_count.extend(Counter(source_words + target_words).most_common())
    target_count.extend(Counter(target_phrases).most_common())

    for word, _ in source_count:
        if word not in source_word2idx:
            source_word2idx[word] = len(source_word2idx)

    for phrase, _ in target_count:
        if phrase not in target_phrase2idx:
            target_phrase2idx[phrase] = len(target_phrase2idx)

    source_data, source_loc_data, target_data, target_label = list(), list(), list(), list()

    # Collect output data (match with source_word2idx) and write to .txt file.
    for sentence in root.iter('sentence'):
        sent = sentence.find('text').text
        sentence_new = re.sub(' +', ' ', sent)
        sptoks = nltk.word_tokenize(sentence_new)
        if len(sptoks) != 0:
            idx = []
            for sptok in sptoks:
                idx.append(source_word2idx[''.join(sptok).lower()])
            for aspectTerms in sentence.iter('aspectTerms'):
                for aspectTerm in aspectTerms.findall('aspectTerm'):
                    if aspectTerm.get("polarity") == "conflict": continue
                    asp = aspectTerm.get('term')
                    if asp != 'NULL':  # Removes implicit targets.
                        asp_new = re.sub(' +', ' ', asp)
                        t_sptoks = nltk.word_tokenize(asp_new)
                        source_data.append(idx)
                        outputtext = ' '.join(sp for sp in sptoks).lower()
                        outputtarget = ' '.join(sp for sp in t_sptoks).lower()
                        outputtext = outputtext.replace(outputtarget, '$T$')
                        out_f.write(outputtext)
                        out_f.write("\n")
                        out_f.write(outputtarget)
                        out_f.write("\n")
                        pos_info, lab = _get_data_tuple(sptoks, t_sptoks, aspectTerm.get('polarity'))
                        pos_info = [(1 - (i / len(idx))) for i in pos_info]
                        source_loc_data.append(pos_info)
                        targetdata = ' '.join(sp for sp in t_sptoks).lower()
                        target_data.append(target_phrase2idx[targetdata])
                        target_label.append(lab)
                        out_f.write(str(lab))
                        out_f.write("\n")

    out_f.close()
    print("Read %s aspects from %s" % (len(source_data), in_file))
    print("Conflicts: " + str(count_confl))
    return source_data, source_loc_data, target_data, target_label, max_sent_len, source_loc_data, max_target_len
