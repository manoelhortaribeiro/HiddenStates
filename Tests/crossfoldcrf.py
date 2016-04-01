from pystruct.learners import NSlackSSVM, LatentSSVM
from Models.GraphLDCRF import GraphLDCRF
from Util.data_parser import load_data
from measures import *
import multiprocessing
import functools
import re
import time
# ------ IO -------


def write_out(mat, results, number_folds, c=1):
    dest = str(re.sub('[.]*/[a-zA-Z0-9_]*/', '', mat))
    dest = str(re.sub('/', '', dest))
    dest = str(re.sub('.mat', '', dest))

    print "../Output/Results/" + dest + '.txt'

    f = open("../Output/Results/" + dest + "c" + str(c) + '.txt', "a+")

    f.write('Dataset = ' + dest + ' Number Folds = ' + str(number_folds) + '\n')

    for idx, items in results.items():

        f.write('n_states: ' + str(idx) + '\n')

        test_opt, train_opt = [], []
        test_sopt, train_sopt = [], []

        for i in items:
            test_opt.append(i[0][0])
            train_opt.append(i[0][1])
            test_sopt.append(i[1][0])
            train_sopt.append(i[1][1])

        avg_test_opt, avg_train_opt = np.array(test_opt).mean(), np.array(train_opt).mean()
        avg_test_sopt, avg_train_sopt = np.array(test_sopt).mean(), np.array(train_sopt).mean()

        std_test_opt, std_train_opt = np.array(test_opt).std(), np.array(train_opt).std()
        std_test_sopt, std_train_sopt = np.array(test_sopt).std(), np.array(train_sopt).std()

        f.write('opti -- Test: ' + str(avg_test_opt) + '+-' + str(std_test_opt) + '\n')
        f.write('     -- Train: ' + str(avg_train_opt) + '+-' + str(std_train_opt) + '\n')
        f.write('sopt -- Test: ' + str(avg_test_sopt) + '+-' + str(std_test_sopt) + '\n')
        f.write('     -- Train: ' + str(avg_train_sopt) + '+-' + str(std_train_sopt) + '\n')

    f.write('--------------------------------------------------------\n')
    f.close()

    return (avg_test_opt, avg_test_sopt)

# ------ Functions ------


def test_states(states, x, y, x_t, y_t, jobs):

    latent_pbl = GraphLDCRF(n_states_per_label=states, inference_method='ad3')

    base_ssvm = NSlackSSVM(latent_pbl, C=1, tol=.01, inactive_threshold=1e-3, batch_size=10, verbose=0, n_jobs=jobs)
    latent_svm = LatentSSVM(base_ssvm=base_ssvm, latent_iter=3)
    latent_svm.fit(x, y)

    test = latent_svm.score(x_t, y_t)
    train = latent_svm.score(x, y)

    print states, 'Test:', test, 'Train:', train
    return test, train


def process_fold(i, x_all, y_all, number_folds, number_states, dist, labels, n_jobs, c):
    testindex = list(range(i, len(x_all), number_folds))
    trainindex = list(set(range(len(x_all))) - set(testindex))

    x_t, y_t = np.array(x_all)[testindex], np.array(y_all)[testindex]
    x, y = np.array(x_all)[trainindex], np.array(y_all)[trainindex]

    prop = calculate_dist(y, x, dist)
    print "Distances calculated"

    optimal_states = divide_hidden_states_measure_c(number_states, labels, prop, c, y)
    suboptimal_states = divide_hidden_states_arbitrary(number_states, labels)

    optimal_result = test_states(optimal_states, x, y, x_t, y_t, jobs=n_jobs)
    suboptimal_result = test_states(suboptimal_states, x, y, x_t, y_t, jobs=n_jobs)

    return optimal_result, suboptimal_result


# ---- Main stuff ------


def validation(mat, x, y, dist, labels, number_folds, states, n_jobs, c):
    our_results, normal_results = dict(), dict()

    for number_states in states:
        results = dict()
        evaluate_fold = functools.partial(process_fold, x_all=x, y_all=y, number_folds=number_folds,
                                          number_states=number_states, labels=labels,
                                          dist=dist, n_jobs=1, c=c)

        p = multiprocessing.Pool(n_jobs)
        results[number_states] = map(evaluate_fold, range(number_folds))
        tmp = write_out(mat, results, number_folds, c)
        our_results[number_states] = tmp[0]
        normal_results[number_states] = tmp[1]

    # Finds best values
    our_states = max(our_results, key=lambda i: our_results[i])
    normal_states = max(normal_results, key=lambda i: normal_results[i])

    return our_states, normal_states


def test(mat, our_states, normal_states, x, y, x_t, y_t, labels, dist, n_jobs, c):

    prop = calculate_dist(y, x, dist)
    print "Distances calculated"

    optimal_states = divide_hidden_states_measure_c(our_states, labels, prop, c, y)
    suboptimal_states = divide_hidden_states_arbitrary(normal_states, labels)

    optimal_result = test_states(optimal_states, x, y, x_t, y_t, jobs=n_jobs)
    suboptimal_result = test_states(suboptimal_states, x, y, x_t, y_t, jobs=n_jobs)

    results = dict()
    results["test_our" + str(our_states) + "normal" + str(normal_states)] = [(optimal_result, suboptimal_result)]
    write_out(mat, results, 0, c)

    return optimal_result, suboptimal_result


def cross_fold_ldcrf(mat, dist, labels, number_folds, states, n_jobs, c=1):
    x, y = load_data(mat)
    results = {}
    results["final"] = []

    for i in range(number_folds):
        test_index = list(range(i, len(x), number_folds))
        validation_index = list(set(range(len(x))) - set(test_index))

        # Loads and split data
        x_v, y_v = np.array(x)[validation_index], np.array(y)[validation_index]
        x_t, y_t = np.array(x)[test_index], np.array(y)[test_index]

        # Does the validation
        our_states, normal_states = validation(mat, x_v, y_v, dist, labels, number_folds, states, n_jobs, c)
        # Does the test
        results["final"].append(test(mat, our_states, normal_states, x_v, y_v, x_t, y_t, labels, dist, n_jobs, c))

        time.sleep(1)

    write_out(mat, results, 0, c)
