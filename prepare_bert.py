# Prepares BERT train and test files with embeddings. Get raw data and BERT embedding first from raw_data and getBERT
# (jupyter notebook files).
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

from config import *

# <editor-fold desc="Combining embedding files, retrieved with 'getBERT">

# Domain is one of the following: restaurant (2014), laptop (2014), book (2019), hotel (2015), Apex (2004),
# Camera (2004), Creative (2004), Nokia (2004).

domain = "restaurant"
year = 2014
path = "data/programGeneratedData/BERT/" + domain + "/"
temp_path = "data/programGeneratedData/BERT/"
split = False


temp_filenames_base = ["data/externalData/" + "BERT_base_" + domain + "_" + str(year) + ".txt"]

FLAGS.source_domain = domain
FLAGS.target_domain = domain
FLAGS.source_year = year

count_sentences = 0
with open(temp_path + "temp/BERT_base_" + str(FLAGS.source_year) + "embedding.txt", 'w', encoding='utf-8') as outf:
    for tfname in temp_filenames_base:
        print(tfname)
        with open(tfname, 'r', encoding='utf-8') as infile:
            for line in infile:
                if line.startswith("\n") or line.startswith("[CLS]") or line.startswith("[SEP]"):
                    pass
                else:
                    outf.write(line)
                    count_sentences += 1
print("First sentences: " + str(count_sentences))
count_sentences = 0
with open(temp_path + "temp/BERT_base_" + str(FLAGS.source_year) + "embedding_withCLS_SEP.txt", 'w') as outf:
    for tfname in temp_filenames_base:
        print(tfname)
        with open(tfname, 'r', encoding='utf-8') as infile:
            for line in infile:
                if line.startswith("\n"):
                    pass
                else:
                    outf.write(line)
                    count_sentences += 1
print("Second sentences: " + str(count_sentences))
# </editor-fold>

# <editor-fold desc="make table with unique words">
vocaBERT = []
vocaBERT_SEP = []
unique_words = []
unique_words_index = []
with open(temp_path + "temp/BERT_base_" + str(FLAGS.source_year) + "embedding_withCLS_SEP.txt") as BERTemb_sep:
    for line in BERTemb_sep:
        word = line.split("_")[0]
        if "[CLS]" in word:
            pass
        else:
            vocaBERT_SEP.append(word)
            if "[SEP]" in word:
                pass
            else:
                if word not in unique_words:
                    unique_words.append(word)
                    unique_words_index.append(0)
                vocaBERT.append(word)
print("vocaBERT: " + str(len(vocaBERT)))
print("vocaBERT_SEP: " + str(len(vocaBERT_SEP)))
# </editor-fold>

# <editor-fold desc="make embedding matrix with unique words, prints counter">
counter = 0
uniqueVocaBERT = []
with open(temp_path + "temp/BERT_base_" + str(FLAGS.source_year) + "embedding.txt", 'r', encoding='utf-8') as BERTemb:
    with open("data/programGeneratedData/" + str(FLAGS.embedding_type) + '_' + domain + "_"
              + str(FLAGS.source_year) + '_' + str(FLAGS.embedding_dim) + '.txt', 'w') as outfile:
        for line in BERTemb:
            word = line.split("_")[0]
            counter += 1
            weights = line.split(" ")[1:]
            try:
                index = unique_words.index(word)  # Get index in unique words table.
                word_count = unique_words_index[index]
                unique_words_index[index] += 1
            except:
                print(word)
            item = str(word) + '_' + str(word_count)
            outfile.write("%s " % item)
            uniqueVocaBERT.append(item)
            first = True
            for weight in weights[:-1]:
                outfile.write("%s " % weight)
            outfile.write("%s" % weights[-1])
# </editor-fold>,
# We now have the embedding matrix with all the unique words.

# <editor-fold desc="make uniqueBERT_SEP variable">
uniqueVocaBERT_SEP = []
counti = 0
for i in range(0, len(vocaBERT_SEP)):
    if vocaBERT_SEP[i] == '[SEP]':
        uniqueVocaBERT_SEP.append('[SEP]')
    else:
        uniqueVocaBERT_SEP.append(uniqueVocaBERT[counti])
        counti += 1
print("vocaBERT_SEP: " + str(len(vocaBERT_SEP)))
print("uniqueVocaBERT: " + str(len(uniqueVocaBERT)))
print("uniqueVocaBERT_SEP: " + str(len(uniqueVocaBERT_SEP)))
# </editor-fold

# <editor-fold desc="make a matrix (three vectors) containing for each word in bert-tokeniser style:
#   word_id (x_word), sentence_id (x_sent), target boolean, (x_targ)">
lines = open(path + "raw_data_" + domain + "_" + str(FLAGS.source_year) + ".txt", encoding="utf-8").readlines()
index = 0
index_sep = 0
x_word = []
x_sent = []
x_targ = []
x_tlen = []
sentenceCount = 0
target_raw = []
sentiment = []
targets_insent = 0
for i in range(0, len(lines), 3):
    target_raw.append(lines[i + 1].lower().split())
    sentiment.append(lines[i + 2])
for i in range(0, len(vocaBERT_SEP)):
    try:
        sentence_target = target_raw[sentenceCount]
    except:
        print(i, sentenceCount, vocaBERT_SEP[i])
    sentence_target_str = ''.join(sentence_target)
    x_word.append(i)
    word = vocaBERT_SEP[i]
    x_sent.append(sentenceCount)
    x_tlen.append(len(sentence_target))
    if word == "[SEP]":
        sentenceCount += 1
        i_new_sent = i + 1
    tar_guess = ""
    for j in range(len(sentence_target) - 1, -1, -1):
        if vocaBERT_SEP[i - j][:2] == '##':
            tar_guess += vocaBERT_SEP[i - j][2:]
        else:
            tar_guess += vocaBERT_SEP[i - j]
    if tar_guess == sentence_target_str:
        x_targ.append(1)
        for k in range(0, len(sentence_target)):
            x_targ[i - k] = 1
    else:
        x_targ.append(0)
# </editor-fold>

# <editor-fold desc="print to BERT data to text file">
for filenr in range(1, 8):
    sentence_senten_unique = ""
    sentence_target_unique = ""
    sentCount = 0
    dollarcount = 0
    with open(temp_path + "temp/" + "unique" + str(FLAGS.source_year) + "_BERT_Data_" + str(filenr) + '.txt', 'w',  encoding='utf-8') as outFile:
        for u in range(0, len(uniqueVocaBERT_SEP)):
            if uniqueVocaBERT_SEP[u] == "[SEP]":
                outFile.write(sentence_senten_unique + '\n')
                outFile.write(sentence_target_unique + '\n')
                outFile.write(''.join(sentiment[sentCount]))
                sentence_senten_unique = ""
                sentence_target_unique = ""
                sentCount += 1
            else:
                if x_targ[u] == 1:
                    dollarcount += 1
                    if dollarcount == 1:
                        sentence_senten_unique += "$T$ "
                    sentence_target_unique += uniqueVocaBERT_SEP[u] + ' '
                else:
                    dollarcount = 0
                    sentence_senten_unique += uniqueVocaBERT_SEP[u] + ' '
    # </editor-fold>

    lines = open(path + "raw_data_" + domain + "_" + str(FLAGS.source_year) + ".txt", encoding="utf-8").readlines()
    index = 0
    index_sep = 0
    x_word = []
    x_sent = []
    x_targ = []
    x_tlen = []
    sentenceCount = 0
    target_raw = []
    sentiment = []
    targets_insent = 0
    for i in range(0, len(lines), 3):
        target_raw.append(lines[i + 1].lower().split())
        sentiment.append(lines[i + 2])
    for i in range(0, len(vocaBERT_SEP)):
        try:
            sentence_target = target_raw[sentenceCount]
        except:
            print(i, sentenceCount, vocaBERT_SEP[i])
        sentence_target_str = ''.join(sentence_target)
        x_word.append(i)
        word = vocaBERT_SEP[i]
        x_sent.append(sentenceCount)
        x_tlen.append(len(sentence_target))
        if word == "[SEP]":
            sentenceCount += 1
            i_new_sent = i + 1
        tar_guess = ""
        for j in range(len(sentence_target) - 1 + filenr, -1, -1):
            if vocaBERT_SEP[i - j][:2] == '##':
                tar_guess += vocaBERT_SEP[i - j][2:]
            else:
                tar_guess += vocaBERT_SEP[i - j]
        if tar_guess == sentence_target_str:
            x_targ.append(1)
            for k in range(0, len(sentence_target) + filenr):
                x_targ[i - k] = 1
        else:
            x_targ.append(0)

# <editor-fold desc="Combine words, this is needed for different tokenisation for target phrase">
# Different files for different extra target lengths, e.g. file 2 contains target phrases that are 1 word longer in the
# BERT embedding than the original target phrase
lines_1 = open(temp_path + "temp/" + "unique" + str(
    FLAGS.source_year) + "_BERT_Data_1.txt",  'r', encoding='utf-8').readlines()
lines_2 = open(temp_path + "temp/" + "unique" + str(
    FLAGS.source_year) + "_BERT_Data_2.txt",  'r', encoding='utf-8').readlines()
lines_3 = open(
    temp_path + "temp/" + "unique" + str(FLAGS.source_year) + "_BERT_Data_3.txt",  'r', encoding='utf-8').readlines()
lines_4 = open(temp_path + "temp/" + "unique" + str(
    FLAGS.source_year) + "_BERT_Data_4.txt",  'r', encoding='utf-8').readlines()
lines_5 = open(temp_path + "temp/" + "unique" + str(FLAGS.source_year) + "_BERT_Data_5.txt",  'r', encoding='utf-8').readlines()
lines_6 = open(temp_path + "temp/" + "unique" + str(FLAGS.source_year) + "_BERT_Data_6.txt",  'r', encoding='utf-8').readlines()
lines_7 = open(temp_path + "temp/" + "unique" + str(FLAGS.source_year) + "_BERT_Data_7.txt",  'r', encoding='utf-8').readlines()
with open(temp_path + "temp/" + "unique" + str(FLAGS.source_year) + "_BERT_Data_All.txt", 'w',  encoding='utf-8') as outF:
    for i in range(0, len(lines_1), 3):
        if lines_1[i + 1] == '\n':
            if lines_2[i + 1] == '\n':
                if lines_3[i + 1] == '\n':
                    if lines_4[i + 1] == '\n':
                        if lines_5[i + 1] == '\n':
                            if lines_6[i + 1] == '\n':
                                outF.write(lines_7[i])
                                outF.write(''.join(lines_7[i + 1]))
                            else:
                                outF.write(lines_6[i])
                                outF.write(''.join(lines_6[i + 1]))
                        else:
                            outF.write(lines_5[i])
                            outF.write(''.join(lines_5[i + 1]))
                    else:
                        outF.write(lines_4[i])
                        outF.write(''.join(lines_4[i + 1]))
                else:
                    outF.write(lines_3[i])
                    outF.write(''.join(lines_3[i + 1]))
            else:
                outF.write(lines_2[i])
                outF.write(''.join(lines_2[i + 1]))
        else:
            outF.write(lines_1[i])
            outF.write(''.join(lines_1[i + 1]))

        outF.write(lines_1[i + 2])
# </editor-fold>

# <editor-fold desc="Split in train and test file">

linesAllData = open(temp_path + "temp/" + "unique" + str(FLAGS.source_year) + "_BERT_Data_All.txt",  'r', encoding='utf-8').readlines()

if split:
    # Split datasets.
    if domain == "laptop":
        # Laptop train originally has 2313 aspects (6939 lines).
        train_lines = 6750
        split_size = 250
    elif domain == "book":
        # Book train has 2700 aspects (8100 lines).
        train_lines = 8100
        split_size = 300
    elif domain == "hotel":
        # Hotel train has 200 aspects (600 lines).
        train_lines = 600
        split_size = 20
    elif domain == "Apex":
        # Apex train has 250 aspects (750 lines).
        train_lines = 750
        split_size = 25
    elif domain == "Camera":
        # Camera train has 310 aspects (930 lines).
        train_lines = 930
        split_size = 31
    elif domain == "Creative":
        # Creative train has 540 aspects (1620 lines).
        train_lines = 1620
        split_size = 54
    elif domain == "Nokia":
        # Nokia train has 220 aspects (660 lines).
        train_lines = 660
        split_size = 22
    for i in range(0, train_lines, 3 * split_size):
        with open(path + str(FLAGS.embedding_dim) + "_" + domain + "_train_" + str(
                FLAGS.source_year) + "_BERT_" + str(
            round((i + 3 * split_size) / 3)) + ".txt", 'w') as outTrain:
            for j in range(0, i + 3 * split_size):
                outTrain.write(linesAllData[j])
    with open(path + str(FLAGS.embedding_dim) + "_" + domain + "_test_" + str(FLAGS.source_year) + '_BERT.txt',
              'w') as outTest:
        for k in range(train_lines, len(linesAllData)):
            outTest.write(linesAllData[k])
else:
    print(domain)
    # Non-split datasets.
    if domain == "restaurant":
        # Restaurant train has 3600 aspects (10800 lines)
        train_lines = 10800
    elif domain == "laptop":
        train_lines = 6750
    elif domain == "book":
        train_lines = 8100
    else:
        # Use 80-20 split for train and test set.
        train_aspects = int(0.8 * (linesAllData / 3))
        train_lines = 3 * train_aspects
    with open(
            path + str(FLAGS.embedding_dim) + "_" + domain + "_train_" + str(FLAGS.source_year) + '_BERT.txt',
            'w') as outTrain:
        for j in range(0, train_lines):
            outTrain.write(linesAllData[j])
    with open(path + str(FLAGS.embedding_dim) + "_" + domain + "_test_" + str(FLAGS.source_year) + '_BERT.txt',
              'w') as outTest:
        for j in range(train_lines, len(linesAllData)):
            outTest.write(linesAllData[j])
# </editor-fold>
