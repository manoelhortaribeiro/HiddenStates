import numpy as np
from pystruct.models import GraphCRF, LatentGraphCRF
from sklearn.cluster import KMeans
from scipy import sparse

__author__ = 'Manoel Ribeiro'


class GraphLDCRF(LatentGraphCRF):
    """LDCRF with latent states for variables.

    This is also called "hidden dynamics CRF".
    For each output variable there is an additional variable which
    can encode additional states and interactions.

    Parameters
    ----------
    n_labels : int
        Number of states of output variables.

    n_features : int or None (default=None).
        Number of input features per input variable.
        ``None`` means it is equal to ``n_labels``.

    n_states_per_label : int or list (default=2)
        Number of latent states associated with each observable state.
        Can be either an integer, which means the same number
        of hidden states will be used for all observable states, or a list
        of integers of length ``n_labels``.

    inference_method : string, default="ad3"
        Function to call to do inference and loss-augmented inference.
        Possible values are:

            - 'max-product' for max-product belief propagation.
                Recommended for chains an trees. Loopy belief propagatin in case of a general graph.
            - 'lp' for Linear Programming relaxation using cvxopt.
            - 'ad3' for AD3 dual decomposition.
            - 'qpbo' for QPBO + alpha expansion.
            - 'ogm' for OpenGM inference algorithms.
    """

    def __init__(self, n_labels=None, n_features=None, n_states_per_label=None,
                 inference_method=None):
        self.n_labels = n_labels
        self.n_states_per_label = n_states_per_label
        GraphCRF.__init__(self, n_states=None, n_features=n_features,
                          inference_method=inference_method)

    def random_init(self, X, Y):
        H = [np.random.randint(self.n_states, size=y.shape) for y in Y]
        return H

    def init_latent(self, X, Y):
        # treat all edges the same
        edges = [[self._get_edges(x)] for x in X]
        features = np.array([self._get_features(x) for x in X])
        return self.random_init(X, Y)



