import multiprocessing
import functools

from pystruct.learners import NSlackSSVM, LatentSSVM

# Internal Imports
from Util.data_parser import load_data, remove_activity_data
from Models.GraphLDCRF import GraphLDCRF
from measures import *


__author__ = 'Manoel Ribeiro'


# does the work for one of the hidden states distributions
def test_case(number_states, s_ent, labels, x, y, x_t, y_t, kind, subopt, opt, svmiter, seed):

    # Gets the different states divisions, notice that the parameter kind, given as input
    # can create a cap, where no label may receive more than k% of the available hidden states
    optimal_states = divide_hidden_states_measure_c(number_states, labels, s_ent, kind, y)

    suboptimal_states = divide_hidden_states_arbitrary(number_states, labels)

    np.random.seed(seed)

    print "Suboptimal States: ", suboptimal_states
    print "Optimal States: ", optimal_states

    # Suppose that we can use 30 hidden states, a naive approach
    # would be to distribute them equally throughout the labels.

    if subopt:
        latent_pbl = GraphLDCRF(n_states_per_label=suboptimal_states, inference_method='dai')
        base_ssvm = NSlackSSVM(latent_pbl, C=1, tol=.01, inactive_threshold=1e-3,
                               batch_size=10, verbose=0, n_jobs=1)
        latent_svm = LatentSSVM(base_ssvm=base_ssvm, latent_iter=svmiter)
        latent_svm.fit(x, y)

        print "------- TEST 1 SUBOPTIMAL STATES -------"
        print("Train: {:2.6f}".format(latent_svm.score(x, y)))
        print("Test: {:2.6f}".format(latent_svm.score(x_t, y_t)))

        sopt_test = latent_svm.score(x_t, y_t)
        sopt_train = latent_svm.score(x, y)

    else:
        sopt_test = 0
        sopt_train = 0

    np.random.seed(seed)

    # Now we go for the sample entropy approach.

    if opt:
        latent_pbl = GraphLDCRF(n_states_per_label=optimal_states, inference_method='dai')
        base_ssvm = NSlackSSVM(latent_pbl, C=1, tol=.01, inactive_threshold=1e-3,
                               batch_size=10, verbose=0, n_jobs=1)
        latent_svm = LatentSSVM(base_ssvm=base_ssvm, latent_iter=svmiter)
        latent_svm.fit(x, y)

        print "------- TEST 2 OPTIMAL STATES -------"
        print("Train: {:2.6f}".format(latent_svm.score(x, y)))
        print("Test: {:2.6f}".format(latent_svm.score(x_t, y_t)))

        opt_test = latent_svm.score(x_t, y_t)
        opt_train = latent_svm.score(x, y)

    else:
        opt_test = 0
        opt_train = 0

    return opt_test, opt_train, sopt_test, sopt_train


# does the work for one fold
def fold_results(tests, labels, datatrain, seqtrain, datatest, seqtest, kind, subopt, opt, svmiter, seed, n_jobs):

    print "Loading data..."

    X, Y, X_t, Y_t = load_data(datatrain, seqtrain, datatest, seqtest)

    print "Data loaded!"

    # print "Removing data..."
    # remove activity from data
    # X, Y = remove_activity_data(X, Y, 9)
    # remove activity from data
    # X_t, Y_t = remove_activity_data(X_t, Y_t, 9)

    print "Calculating Sample Entropy..."
    s_ent = sample_entropy(X, Y)
    print "Sample Entropy: ", s_ent

    # Arrays containing the results.

    # ~Optimal~ stuff
    opt_tests = []
    opt_trains = []

    # Suboptimal stuff
    sopt_tests = []
    sopt_trains = []

    print "Starting test!"

    evaluate = functools.partial(test_case, s_ent=s_ent, labels=labels, x=X, y=Y, x_t=X_t, y_t=Y_t,
                                 kind=kind, subopt=subopt, opt=opt, svmiter=svmiter, seed=seed)

    p = multiprocessing.Pool(n_jobs)
    t = p.map(evaluate, tests)

    for i in t:

        opt_tests.append(i[0])
        opt_trains.append(i[1])

        sopt_tests.append(i[2])
        sopt_trains.append(i[3])

    return opt_tests, opt_trains, sopt_tests, sopt_trains


# does all the folds in a data-set
def eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold,
                  kind=1, subopt=True, opt=True, svmiter=10, seed=1, n_jobs=4):

    opt_tests = []
    opt_trains = []

    sopt_tests = []
    sopt_trains = []

    # evaluates one fold at a time
    for i in folds:

        print "FOLD:", i

        # test
        dte = path + data + test + name + fold + str(i) + ".csv"
        sqte = path + label + test + name + fold + str(i) + ".csv"

        # train
        dtr = path + data + train + name + fold + str(i) + ".csv"
        sqtr = path + label + train + name + fold + str(i) + ".csv"

        opt_test, opt_train, sopt_test, sopt_train = fold_results(tests, n_labels, dtr, sqtr, dte,
                                                                  sqte, kind, subopt, opt, svmiter,
                                                                  seed, n_jobs)

        opt_tests.append(opt_test)
        opt_trains.append(opt_train)

        sopt_tests.append(sopt_test)
        sopt_trains.append(sopt_train)

    # Transformation into numpy array

    opt_tests = np.array(opt_tests)
    opt_trains = np.array(opt_trains)
    sopt_tests = np.array(sopt_tests)
    sopt_trains = np.array(sopt_trains)

    # Optimal Data

    opt_tests_avg_std = (opt_tests.mean(0), opt_tests.std(0))
    opt_trains_avg_std = (opt_trains.mean(0), opt_trains.std(0))

    # Suboptimal Data

    sopt_tests_avg_std = (sopt_tests.mean(0), sopt_tests.std(0))
    sopt_trains_avg_std = (sopt_trains.mean(0), sopt_trains.std(0))

    return opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std

