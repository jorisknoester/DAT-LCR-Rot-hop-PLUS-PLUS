# Hyperparameter tuning using Tree Parzen Estimator (TPE).
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

import json
import os
import pickle
from functools import partial

from bson import json_util
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK

import DAT_LCR_Rot_hop_plus_plus
from config import *
from load_data import *

global eval_num, best_loss, best_hyperparams


def main():
    """
    Runs hyperparameter tuning for each model specified in domains.

    :return:
    """
    runs = 10
    n_iter = 20

    # Name, year of source, name, year of target, corresponding batch sizes as defined in main_test
    rest_lapt = ["restaurant", 2014, "laptop", 2014, 24, 15, 701]
    rest_book = ["restaurant", 2014, "book", 2019, 24, 18, 804]
    lapt_rest = ["laptop", 2014, "restaurant", 2014, 20, 24, 1122]
    lapt_book = ["laptop", 2014, "book", 2019, 20, 24, 804]
    book_rest = ["book", 2019, "restaurant", 2014, 18, 24, 1122]
    book_lapt = ["book", 2019, "laptop", 2014, 24, 20, 701]

    domains = [rest_lapt, rest_book, lapt_rest, lapt_book, book_rest, book_lapt]

    for domain in domains:
        run_hyper(source_domain=domain[0], year_source=domain[1], target_domain=domain[2], year_target=domain[3],
                  batch_size_src=domain[4], batch_size_tar=domain[5], batch_size_te=domain[6], runs=runs, n_iter=n_iter)


def run_hyper(source_domain, year_source, target_domain, year_target, batch_size_src, batch_size_tar, batch_size_te,
              runs, n_iter):
    """
    Runs hyperparameter tuning for the specified domain.

    :param source_domain: source domain name
    :param year_source: source domain year
    :param target_domain: target domain name
    :param year_target: target domain year
    :param batch_size_src: batch size source domain
    :param batch_size_tar: batch size target domain
    :param batch_size_te: total sample size of test set
    :param runs: the number of hyperparameter tuning runs
    :param n_iter: the number of iterations for each hyperparameter tuning run
    :return:
    """
    path = "hyper_results/CLRH/" + source_domain + "/" + str(target_domain) + "/"

    FLAGS.source_domain = source_domain
    FLAGS.source_year = source_year
    FLAGS.target_domain = target_domain
    FLAGS.target_year = target_year
    FLAGS.batch_size_src = batch_size_src
    FLAGS.batch_size_tar = batch_size_tar
    FLAGS.batch_size_te = batch_size_te
    FLAGS.n_iter = n_iter

    FLAGS.train_data_source = "data/externalData/" + FLAGS.source_domain + "_train_" + str(FLAGS.source_year) + ".xml"
    FLAGS.train_data_target = "data/externalData/" + FLAGS.target_domain + "_train_" + str(FLAGS.target_year) + ".xml"
    FLAGS.test_data = "data/externalData/" + FLAGS.target_domain + "_test_" + str(FLAGS.target_year) + ".xml"
    FLAGS.train_path_source = "data/programGeneratedData/BERT/" + FLAGS.source_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.source_domain + "_train_" + str(source_year) + "_BERT.txt"
    FLAGS.train_path_target = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_train_" + str(target_year) + "_BERT.txt"
    FLAGS.test_path = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_test_" + str(FLAGS.target_year) + "_BERT.txt"
    FLAGS.train_embedding_source = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.source_domain + "_" + str(
        source_year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.train_embedding_target = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
        target_year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.test_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.target_year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.hyper_train_path_source = "data/programGeneratedData/" + str(
        FLAGS.embedding_dim) + 'hypertraindatasource' + "_" + str(
        FLAGS.source_domain) + "_" + str(FLAGS.target_domain) + ".txt"
    FLAGS.hyper_train_path_target = "data/programGeneratedData/" + str(
        FLAGS.embedding_dim) + 'hypertraindatatarget' + "_" + str(
        FLAGS.source_domain) + "_" + str(FLAGS.target_domain) + ".txt"
    FLAGS.hyper_eval_path_target = "data/programGeneratedData/" + str(
        FLAGS.embedding_dim) + 'hyperevaldatatarget' + "_" + str(
        FLAGS.source_domain) + "_" + str(FLAGS.target_domain) + ".txt"
    FLAGS.results_file = "hyper_results/CLRH/" + source_domain + "/" + str(target_domain) + "/hyperresults.txt"

    # Get statistics from newly generated hyperresults files.
    _, _, _, _, _, _ = load_hyper_data(FLAGS, shuffle=True)

    # Define variable spaces for hyperparameter optimization to run over.
    global eval_num, best_loss, best_hyperparams
    eval_num = 0
    best_loss = None
    best_hyperparams = None

    CLRH_space = [
        hp.choice('learning_rate_dis', [0.03, 0.01, 0.005]),
        hp.choice('learning_rate_f', [0.03, 0.01, 0.005]),
        hp.choice('keep_prob', [0.25, 0.30, 0.35]),
        hp.choice('momentum_dis', [0.80, 0.85, 0.90]),
        hp.choice('momentum_f', [0.80, 0.85, 0.90]),
        hp.choice('l2_dis', [0.01, 0.001, 0.0001]),
        hp.choice('l2_f', [0.01, 0.001, 0.0001]),
        hp.choice('balance_lambda', [0.6, 0.8, 1.1])
    ]

    for i in range(runs):
        print("Optimizing New Model\n")
        run_a_trial(CLRH_space, path)
        plot_best_model(path)


def CLRH_objective(hyperparams, path):
    """
    Compares the losses of the different hyperparameter optimisations and decides which setting is best.

    :param hyperparams: hyperparameters (learning rate_dis, learning_rate_f, keep probability, momentum_dis, momentum_f,
     l2-regularization term, and balance parameter lambda)
    :param path: save path
    :return: the result to be written to the results file.
    """
    global eval_num, best_loss, best_hyperparams

    eval_num += 1
    (learning_rate_dis, learning_rate_f, keep_prob, momentum_dis, momentum_f, l2_dis, l2_f, balance_lambda) = hyperparams
    print("Current hyperparameters: " + str(hyperparams))

    l, pred1, fw1, bw1, tl1, tr1 = DAT_LCR_Rot_hop_plus_plus.main(FLAGS.hyper_train_path_source, FLAGS.hyper_train_path_target,
                                            FLAGS.hyper_eval_path_target, learning_rate_dis, learning_rate_f,
                                            keep_prob, momentum_dis, momentum_f, l2_dis, l2_f, balance_lambda)
    tf.reset_default_graph()

    if best_loss is None or -l < best_loss:
        best_loss = -l
        best_hyperparams = hyperparams

    result = {
        'loss': -l,
        'status': STATUS_OK,
        'space': hyperparams,
    }

    save_json_result(str(l), result, path)

    return result


def run_a_trial(CLRH_space, path):
    """
    Runs the model once for the specified hyperparameter settings and defined iterations.

    :param CLRH_space: tuning space for CLRH model
    :param path: save path
    :return:
    """
    max_evals = nb_evals = 1

    print("Attempt to resume a past training if it exists:")

    try:
        trials = pickle.load(open(path + "results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    objective = CLRH_objective
    partial_objective = partial(objective, path=path)
    space = CLRH_space

    best = fmin(
        fn=partial_objective,
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open(path + "results.pkl", "wb"))

    print("OPTIMIZATION STEP COMPLETE.\n")


def print_json(result):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param result:
    :return:
    """
    """Pretty-print a jsonable structure (e.g.: result)."""
    print(json.dumps(
        result,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    ))


def save_json_result(model_name, result, path):
    """
    Save json to a directory and a filename. Method obtained from Trusca et al. (2020).

    :param model_name:
    :param result:
    :param path:
    :return:
    """
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )


def load_json_result(best_result_name, path):
    """
    Load json from a path (directory + filename). Method obtained from Trusca et al. (2020).

    :param best_result_name:
    :param path:
    :return:
    """
    result_path = os.path.join(path, best_result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(
            f.read()
        )


def load_best_hyperspace(path):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param path:
    :return:
    """
    results = [
        f for f in list(sorted(os.listdir(path))) if 'json' in f
    ]
    if len(results) == 0:
        return None

    best_result_name = results[-1]
    return load_json_result(best_result_name, path)["space"]


def plot_best_model(path):
    """
    Plot the best model found yet. Method obtained from Trusca et al. (2020).

    :param path:
    :return:
    """
    space_best_model = load_best_hyperspace(path)
    if space_best_model is None:
        print("No best model to plot. Continuing...")
        return

    print("Best hyperspace yet:")
    print_json(space_best_model)


if __name__ == "__main__":
    main()
