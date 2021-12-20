# Method for preparing raw data for the restaurant, laptop, book and hotel datasets.
#
# https://github.com/jorisknoester/DAT-LCR-Rot-hop-PLUS-PLUS
#
# Adapted from van Berkum, van Megen, Savelkoul, and Weterman (2021).
# https://github.com/stefanvanberkum/CD-ABSC
#
# van Berkum, S., van Megen, S., Savelkoul, M., and Weterman, P. (2021) Fine-Tuning for Cross-Domain
# Aspect-Based Sentiment Classification. Theoretical report Erasmus University Rotterdam.

from data_book_hotel import read_book_hotel
from data_rest_lapt import read_rest_lapt


def main():
    """
    Gets the raw data for the specified domain.

    :return:
    """
    # Domain is one of the following: restaurant (2014), laptop (2014), book (2019), hotel (2015).

    domain = "restaurant"
    year = 2014

    if domain == "restaurant" or domain == "laptop":
        train_file = "data/externalData/" + domain + "_train_" + str(year) + ".xml"
        test_file = "data/externalData/" + domain + "_test_" + str(year) + ".xml"
        train_out = "data/programGeneratedData/BERT/" + domain + "/raw_data_" + domain + "_train_" + str(year) + ".txt"
        test_out = "data/programGeneratedData/BERT/" + domain + "/raw_data_" + domain + "_test_" + str(year) + ".txt"
        with open(train_out, "w") as out:
            out.write("")
        with open(test_out, "w") as out:
            out.write("")
        read_rest_lapt(in_file=train_file, source_count=[], source_word2idx={}, target_count=[], target_phrase2idx={},
                       out_file=train_out)
        read_rest_lapt(in_file=test_file, source_count=[], source_word2idx={}, target_count=[], target_phrase2idx={},
                       out_file=test_out)
    else:
        in_file = "data/externalData/" + domain + "_reviews_" + str(year) + ".xml"
        out_file = "data/programGeneratedData/BERT/" + domain + "/raw_data_" + domain + "_" + str(year) + ".txt"
        with open(out_file, "w") as out:
            out.write("")
        read_book_hotel(in_file=in_file, source_count=[], source_word2idx={}, target_count=[], target_phrase2idx={},
                        out_file=out_file)


if __name__ == '__main__':
    main()
