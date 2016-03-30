import scipy.spatial.distance as distance

from Tests.crossfoldcrf import cross_fold_ldcrf

__author__ = 'Manoel Ribeiro'

labels = 2
number_folds = 5
states = [2, 4, 6, 8, 10]
n_jobs = 5


# -- Continuous

cross_fold_ldcrf(mat='../Dataset/synthetic.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs)
