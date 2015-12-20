from pystruct.learners import NSlackSSVM, LatentSSVM

from Tests.measures import *

# Internal Imports
from Util.data_parser import load_data
from Models.GraphLDCRF import GraphLDCRF
import multiprocessing
import os
import functools
import random


def load(vdtr, vdte, vstr, vste, tdtr, tdte, tstr, tste, number_folds=3):
    path = os.environ['PYTHONPATH'].split(os.pathsep)[0]

    val_datatrain = path + "/Dataset/Data/" + vdtr
    val_datatest = path + "/Dataset/Data/" + vdte
    val_seqtrain = path + "/Dataset/Data/" + vstr
    val_seqtest = path + "/Dataset/Data/" + vste

    x = [0] * number_folds
    x_t = [0] * number_folds
    y = [0] * number_folds
    y_t = [0] * number_folds

    for i in range(number_folds):
        x[i], y[i], x_t[i], y_t[i] = load_data(val_datatrain + str(i + 1) + ".csv",
                                               val_seqtrain + str(i + 1) + ".csv",
                                               val_datatest + str(i + 1) + ".csv",
                                               val_seqtest + str(i + 1) + ".csv")

    tst_datatrain = path + "/Dataset/Data/" + tdtr
    tst_datatest = path + "/Dataset/Data/" + tdte
    tst_seqtrain = path + "/Dataset/Data/" + tstr
    tst_seqtest = path + "/Dataset/Data/" + tste

    t_x, t_y, t_x_t, t_y_t = load_data(tst_datatrain, tst_seqtrain, tst_datatest, tst_seqtest)

    return x, x_t, y, y_t, t_x, t_x_t, t_y, t_y_t


def evaluate(states, x, y, x_t, y_t):

    number_folds = len(x)

    test = 0
    train = 0

    for _x, _y, _x_t, _y_t in zip(x, y, x_t, y_t):
        latent_pbl = GraphLDCRF(n_states_per_label=states, inference_method='dai')

        base_ssvm = NSlackSSVM(latent_pbl, C=1, tol=.01, inactive_threshold=1e-3,
                               verbose=0, n_jobs=1, batch_size=10)

        latent_svm = LatentSSVM(base_ssvm=base_ssvm, latent_iter=5)
        latent_svm.fit(_x, _y)

        train += latent_svm.score(_x, _y)
        partial_test = latent_svm.score(_x_t, _y_t)
        test += partial_test

    test /= number_folds

    return test


def test(greedy_states, arbitrary_states, x, y, x_t, y_t, numberseeds=3):
    random.seed(1)
    np.random.seed(1)
    test_arb = []
    test_gre = []

    for i in range(numberseeds):
        test_arb.append(evaluate(arbitrary_states, x, y, x_t, y_t))
        test_gre.append(evaluate(greedy_states, x, y, x_t, y_t))

    test_arb = np.array(test_arb).mean()
    test_gre = np.array(test_gre).mean()

    return test_arb, test_gre


def greedy(x, y, x_t, y_t, add=1, rangeof=6, number_seeds=3):
    previous = [0] * rangeof
    init = [1, 1, 1, 1, 1, 1]
    best = [[],0,0] # states, accuracy, iterations without update
    without_upd = 3

    for iter in range(21):
        random.seed(1)
        np.random.seed(1)

        process_pool = multiprocessing.Pool(36)
        func = functools.partial(evaluate, x=x, y=y, x_t=x_t, y_t=y_t)

        test = np.zeros(rangeof*rangeof + rangeof)
        states = []

        for a in range(rangeof):

            tmp = list(init)
            tmp[a] += add
            states.append(tmp)

            for b in range(rangeof):
                tmp = list(init)
                tmp[a] += add
                tmp[b] += add
                states.append(tmp)

        for i in range(number_seeds):
            test += np.array(process_pool.map(func, states))

        test = (test / number_seeds).tolist()
        # print "TEST =", test

        old_winner = previous.index(max(previous))
        winner = test.index(max(test))

        init = list(states[winner])

        if test[winner] > best[1]:
            best[0] = list(init)
            best[1] = test[winner]
            best[2] = 0
        else:
            best[2] += 1

        print init, previous[old_winner], test[winner], "best: ", best[0], best[1]

        previous = list(test)

        if best[2] >= without_upd:
            break

    return best[0], best[1]


def arbitrary(x, y, x_t, y_t, add=1, rangeof=6, number_seeds=3):
    previous = [0] * rangeof
    init = [1, 1, 1, 1, 1, 1]
    best = [[],0,0] # states, accuracy, iterations without update

    np.random.seed(1)

    for iter in range(7):
        func = functools.partial(evaluate, x=x, y=y, x_t=x_t, y_t=y_t)

        test = np.zeros(rangeof)
        states = []

        tmp = list(init)
        for j in range(rangeof):
            tmp[j] += add

        states.append(tmp)

        # print states
        for i in range(number_seeds):
            test += np.array(map(func, states))

        test = (test / number_seeds).tolist()
        # print "TEST =", test

        old_winner = previous.index(max(previous))
        winner = test.index(max(test))

        init = list(states[0])

        if test[winner] > best[1]:
            best[0] = list(init)
            best[1] = test[winner]
            best[2] = 0

        print init, previous[old_winner], test[winner]

        previous = list(test)

    return init, max(previous)
