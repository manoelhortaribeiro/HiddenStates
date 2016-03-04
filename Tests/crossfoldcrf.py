import multiprocessing
import functools
import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# PyStruct
from pystruct.learners import NSlackSSVM, LatentSSVM

# Internal Imports
from Util.data_parser import load_data
from Models.GraphLDCRF import GraphLDCRF
from measures import *
import scipy.spatial.distance as distance


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
        plt.savefig('../Output/Confusion_Matrix/' + dest+str(i)+'.png')


def test_states(i, states, x, y, x_t, y_t, dest):

        latent_pbl = GraphLDCRF(n_states_per_label=states, inference_method='dai')

        base_ssvm = NSlackSSVM(latent_pbl, C=1, tol=.01, inactive_threshold=1e-3, batch_size=10, verbose=0, n_jobs=1)
        latent_svm = LatentSSVM(base_ssvm=base_ssvm, latent_iter=5)
        latent_svm.fit(x, y)

        test = latent_svm.score(x_t, y_t)
        train = latent_svm.score(x, y)

        plot_cm(latent_svm, y_t, x_t, dest, i)

        print states, 'Test:', test, 'Train:', train

        return test, train


def process_fold(i, X, Y, number_folds, number_states, dist, labels, mat):

        dest = str(re.sub('[.]*/([a-zA-Z0-9_]*/)*','',mat)[:-4]) + str(number_states)

        testindex = list(range(i, len(X), number_folds))
        trainindex = list(set(range(len(X))) - set(testindex))

        x_t = np.array(X)[testindex]
        y_t = np.array(Y)[testindex]

        x = np.array(X)[trainindex]
        y = np.array(Y)[trainindex]

        #prop = [(0, 0.1702469489578651), (1, 0.82975305104215569)]
        prop = calculate_dist(y, x, distance.cosine)

        optimal_states = divide_hidden_states_measure_c(number_states, labels, prop, 1, y)
        suboptimal_states = divide_hidden_states_arbitrary(number_states, labels)

        optimal_result = test_states(i, optimal_states, x, y, x_t, y_t, dest)
        suboptimal_result = test_states(i, suboptimal_states, x, y, x_t, y_t, dest)

        return optimal_result, suboptimal_result

def cross_fold_ldcrf(mat, dist=distance.sqeuclidean):

    X, Y = load_data(mat)
    number_folds = 7
    number_states = 5
    labels = 2
    n_jobs = 6

    evaluate_fold = functools.partial(process_fold, X=X, Y=Y, number_folds=number_folds,
                                      number_states=number_states,labels=labels,
                                      mat=mat, dist=dist)

    p = multiprocessing.Pool(n_jobs)
    t = p.map(evaluate_fold, range(number_folds))


