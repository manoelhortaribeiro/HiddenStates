import numpy as np
from pystruct.models import GraphCRF, LatentGraphCRF
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

__author__ = 'Manoel Ribeiro'


def kmeans_init(data, data_grouping):
    pca = PCA(n_components=2).fit(data)
    estimator = KMeans(init=pca.components_, n_clusters=2, n_init=1)
    estimator.fit(data)
    labels = estimator.labels_
    hidden_states = []
    for group in data_grouping:
        hidden_states.append(np.array(labels[:group]))
        labels = labels[group:]

    return hidden_states


class GraphLDCRF(LatentGraphCRF):

    def __init__(self, n_labels=None, n_features=None, n_states_per_label=None,
                 inference_method=None):
        self.n_labels = n_labels
        self.n_states_per_label = n_states_per_label
        GraphCRF.__init__(self, n_states=None, n_features=n_features,
                          inference_method=inference_method)

    def init_latent(self, X, Y):

        data = []
        data_grouping = []
        for time_window in X:
            data_grouping.append(len(time_window[0]))
            for data_point in time_window[0]:
                data.append(list(data_point))

        hidden_states = kmeans_init(data, data_grouping)

        return hidden_states



