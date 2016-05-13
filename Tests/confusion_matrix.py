import scipy.spatial.distance as distance
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pystruct.learners import NSlackSSVM, LatentSSVM

from Models.GraphLDCRF import GraphLDCRF
from Util.data_parser import load_data
from measures import *

__author__ = 'Manoel Ribeiro'


def plot_cm(latent_svm, y_t, x_t, dest, i):
    Y_ts = []
    for Y_ti in y_t:
        Y_ts.append(Y_ti.tolist())

    Y_p = latent_svm.predict(x_t)
    Y_ps = []
    for Y_pi in Y_p:
        Y_ps.append(Y_pi.tolist())

    cm = confusion_matrix(sum(Y_ts, []), sum(Y_ps, []))
    # Show confusion matrix in a separate window
    plt.matshow(cm, cmap=plt.get_cmap('Greys'))

    plt.title('Confusion matrix')
    for i, cas in enumerate(cm):
        for j, c in enumerate(cas):
            if c > 0:
                plt.text(j - .2, i + .2, c, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('../Output/Confusion_Matrix/' + dest + str(i) + '.png')


def test_states(states, x, y, x_t, y_t, i, jobs):
    latent_pbl = GraphLDCRF(n_states_per_label=states, inference_method='qpbo')

    base_ssvm = NSlackSSVM(latent_pbl, C=1, tol=.01, inactive_threshold=1e-3, batch_size=10, verbose=0, n_jobs=jobs)
    latent_svm = LatentSSVM(base_ssvm=base_ssvm, latent_iter=3)
    latent_svm.fit(x, y)

    test = latent_svm.score(x_t, y_t)
    train = latent_svm.score(x, y)

    plot_cm(latent_svm, y_t, x_t, str(states), i)

    print states, 'Test:', test, 'Train:', train
    return test, train


def test(our_states, normal_states, x, y, x_t, y_t, dist, n_jobs, i):
    prop = calculate_dist(y, x, dist)
    print prop

    optimal_states = our_states
    suboptimal_states = normal_states

    optimal_result = test_states(optimal_states, x, y, x_t, y_t, i, jobs=n_jobs)
    suboptimal_result = test_states(suboptimal_states, x, y, x_t, y_t, i, jobs=n_jobs)

    results = dict()
    results["test_our" + str(our_states) + "normal" + str(normal_states)] = [(optimal_result, suboptimal_result)]

    return optimal_result, suboptimal_result


def cross_fold_ldcrf(mat, dist, number_folds, n_jobs, our_states, normal_states):
    x, y = load_data(mat)
    results = dict()
    results["final"] = []

    for i in range(number_folds):
        test_index = list(range(i, len(x), number_folds))
        validation_index = list(set(range(len(x))) - set(test_index))

        # Loads and split data
        x_v, y_v = np.array(x)[validation_index], np.array(y)[validation_index]
        x_t, y_t = np.array(x)[test_index], np.array(y)[test_index]

        # Does the test
        results["final"].append(test(our_states, normal_states, x_v, y_v, x_t, y_t, dist, n_jobs, i))


cross_fold_ldcrf(mat='../Dataset/NATOPS/23_0145c.mat', dist=distance.sqeuclidean,
                 number_folds=3, n_jobs=6,
                 our_states=[2, 1], normal_states=[1, 2])

cross_fold_ldcrf(mat='../Dataset/NATOPS/23_0145c.mat', dist=distance.sqeuclidean,
                 number_folds=3, n_jobs=6,
                 our_states=[1, 1], normal_states=[2, 2])
