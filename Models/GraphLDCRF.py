import numpy as np
from pystruct.models import GraphCRF, LatentGraphCRF
from sklearn.cluster import KMeans
from scipy import sparse

__author__ = 'Manoel Ribeiro'


def kmeans_init(X, Y, all_edges, n_labels, n_states_per_label,
                symmetric=True):
    all_feats = []
    # iterate over samples
    for x, y, edges in zip(X, Y, all_edges):
        # first, get neighbor counts from nodes
        n_nodes = x.shape[0]
        labels_one_hot = np.zeros((n_nodes, n_labels), dtype=np.int)
        y = y.ravel()
        gx = np.ogrid[:n_nodes]
        labels_one_hot[gx, y] = 1

        size = np.prod(y.shape)
        graphs = [sparse.coo_matrix((np.ones(e.shape[0]), e.T), (size, size))
                  for e in edges]
        if symmetric:
            directions = [g + g.T for g in graphs]
        else:
            directions = [T for g in graphs for T in [g, g.T]]
        neighbors = [s * labels_one_hot.reshape(size, -1) for s in directions]
        neighbors = np.hstack(neighbors)
        # normalize (for borders)
        neighbors /= np.maximum(neighbors.sum(axis=1)[:, np.newaxis], 1)

        # add unaries
        features = np.hstack([x, neighbors])
        all_feats.append(features)
    all_feats_stacked = np.vstack(all_feats)
    Y_stacked = np.hstack(Y).ravel()
    # for each state, run k-means over whole dataset
    H = [np.zeros(y.shape, dtype=np.int) for y in Y]
    label_indices = np.hstack([0, np.cumsum(n_states_per_label)])
    for label in np.unique(Y_stacked):
        try:
            km = KMeans(n_clusters=n_states_per_label[label])
        except TypeError:
            # for old versions :-/
            km = KMeans(k=n_states_per_label[label])
        indicator = Y_stacked == label
        f = all_feats_stacked[indicator]
        km.fit(f)
        for feats_sample, y, h in zip(all_feats, Y, H):
            indicator_sample = y.ravel() == label
            pred = km.predict(feats_sample[indicator_sample]).astype(np.int)
            h.ravel()[indicator_sample] = pred + label_indices[label]
    return H

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
        return kmeans_init(features, Y, edges, n_labels=self.n_labels,
                           n_states_per_label=self.n_states_per_label)



