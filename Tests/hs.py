import numpy as np

from sklearn import metrics
from sklearn.metrics import confusion_matrix

from pystruct.learners import NSlackSSVM, LatentSSVM

# Internal Imports
import Util.pyeeg as pyeeg  # Contains the sample entropy calculation
from Util.data_parser import load_data
from Models.GraphLDCRF import GraphLDCRF


__author__ = 'Manoel Ribeiro'


def calculate_maximum_hidden(buckets, y):
        # calculate maximum number of hidden states as number of instances
    maximum = []
    for i in range(buckets):
        maximum.append(0)

    for i in y:
        for j in i:
            maximum[j] += 1

    return maximum


def sample_entropy(X, Y):

    a = {}

    done = 0.0

    for k in range(len(X)):
        for i, j in zip(X[k][0],Y[k]):
            if a.has_key(j) == False:
                a[j] = []
            a[j].append(i)

    samp_entropy_sum = {}

    for key in a.keys():
        matrix = np.array(a[key]).transpose()
        partial = 0

        for i in matrix:
            std = np.std(i)
            partial += pyeeg.samp_entropy(i, 2, 0.2*std)

        samp_entropy_sum[key] = partial

        done += 1.0
        print str(done/len(a.keys())*100) + "% DONE!"

    total_entropy = 0

    for key in a.keys():
        total_entropy += samp_entropy_sum[key]

    for key in a.keys():
        samp_entropy_sum[key] = samp_entropy_sum[key]/(float(total_entropy))

    return samp_entropy_sum.items()


def divide_hidden_states_entropy_c(balls, buckets, measure, c, y):

    maximum = calculate_maximum_hidden(buckets, y)

    original_balls = balls

    # initialize each bucket with one ball
    balls -= buckets
    balls_dist = buckets

    states = []
    for i in range(buckets):
        states.append(1)

    # gets measure into array and normalize it
    values = []
    for i, j in measure:
        values.append(j)

    sum_values = np.array(values).sum()
    values = map(lambda x: x/sum_values, values)

    # get actual values in the distribution
    actual_values = [float(x) / balls_dist for x in states]

    # distribute the rest of the buckets
    while balls != 0:
        diff = [a-b for a, b in zip(values, actual_values)]

        # gets the tuple with (x,y) where X,
        # is position, and Y is the value of diff
        tuplediff = sorted(enumerate(diff), key=(lambda k: k[1]))

        # do some bookkeeping
        balls -= 1
        balls_dist += 1

        for i in range(buckets):
            index = tuplediff[-(i+1)][0]
            if states[index] < c * original_balls or i+1 == buckets:
                if states[index] < maximum[index]:
                    states[index] += 1
                    break

        actual_values = [float(x) / balls_dist for x in states]

    return states


def divide_hidden_states_normal(balls, buckets):

    states = []

    # Initialize each state with 0 balls
    for i in range(buckets):
        states.append(0)

    # Add balls to states in order
    pointer = 0
    for i in range(balls):
        states[pointer % buckets] += 1
        pointer += 1

    return states


# does the work for one of the hidden states distributions
def test_case(number_states, s_ent, labels, x, y, x_t, y_t, kind, subopt, opt, svmiter, seed, n_jobs):

    # Gets the different states divisions
    if kind == "Equal":
        optimal_states = divide_hidden_states_entropy_c(number_states, labels, s_ent, 1, y)
    if kind == "Capped40%":
        optimal_states = divide_hidden_states_entropy_c(number_states, labels, s_ent, 0.4, y)
    if kind == "Capped30%":
        optimal_states = divide_hidden_states_entropy_c(number_states, labels, s_ent, 0.3, y)
    if kind == "Capped20%":
        optimal_states = divide_hidden_states_entropy_c(number_states, labels, s_ent, 0.2, y)

    suboptimal_states = divide_hidden_states_normal(number_states, labels)

    np.random.seed(seed)

    print "Suboptimal States: ", suboptimal_states
    print "Optimal States: ", optimal_states

    # TEST 1 #
    # Suppose that we can use 30 hidden states, a naive approach
    # would be to distribute them equally throughout the labels.

    if subopt:
        latent_pbl = GraphLDCRF(n_states_per_label=optimal_states, inference_method='dai')
        base_ssvm = NSlackSSVM(latent_pbl, C=1, tol=.01,
                               inactive_threshold=1e-3, batch_size=10, verbose=0, n_jobs=n_jobs)
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

    # TEST 2 #
    # Now we go for the sample entropy approach.

    if opt:
        latent_pbl = GraphLDCRF(n_states_per_label=optimal_states, inference_method='dai')
        base_ssvm = NSlackSSVM(latent_pbl, C=1, tol=.01, inactive_threshold=1e-3,
                               batch_size=10, verbose=0, n_jobs=n_jobs)
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
    x, y, x_t, y_t = load_data(datatrain, seqtrain, datatest, seqtest)
    print "Data loaded!"

    print "Calculating Sample Entropy..."

    s_ent = sample_entropy(x, y)
    #s_ent = [(0,0.1), (1,0.1), (2, 0.1), (3, 0.1), (4, 0.1), (5, 0.1), (6, 0.1), (7, 0.1), (8, 0.1), (9, 0.5)]

    print "Sample Entropy Calculated!"

    print "Sample Entropy: ", s_ent

    # Arrays containing the results.

    # ~Optimal~ stuff
    opt_tests = []
    opt_trains = []

    # Suboptimal stuff
    sopt_tests = []
    sopt_trains = []

    print "Starting test!"

    for i in tests:
        opt_test, opt_train, sopt_test, sopt_train = test_case(i, s_ent, labels, x, y, x_t, y_t,
                                                               kind, subopt, opt, svmiter, seed, n_jobs)

        opt_tests.append(opt_test)
        opt_trains.append(opt_train)

        sopt_tests.append(sopt_test)
        sopt_trains.append(sopt_train)

    return opt_tests, opt_trains, sopt_tests, sopt_trains


# does all the folds in a data-set
def eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold,
                  kind="Equal", subopt=True, opt=True, svmiter=10, seed=1, n_jobs=1):

    opt_tests = []
    opt_trains = []

    sopt_tests = []
    sopt_trains = []

    for i in folds:

        print "FOLD:", i
        # test
        dte = path + data + test + name + fold + str(i) + ".csv"
        sqte = path + label + test + name + fold + str(i) + ".csv"

        # train
        dtr = path + data + train + name + fold + str(i) + ".csv"
        sqtr = path + label + train + name + fold + str(i) + ".csv"

        opt_test, opt_train, sopt_test, sopt_train = fold_results(tests, n_labels, dtr, sqtr, dte,
                                                                  sqte, kind, subopt, opt, svmiter, seed, n_jobs)

        opt_tests.append(opt_test)
        opt_trains.append(opt_train)

        sopt_tests.append(sopt_test)
        sopt_trains.append(sopt_train)

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


