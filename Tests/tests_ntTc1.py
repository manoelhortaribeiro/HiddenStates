import scipy.spatial.distance as distance

from Tests.crossfoldcrf import cross_fold_ldcrf

__author__ = 'Manoel Ribeiro'

labels = 6
number_folds = 5
states = [6, 12, 18, 24, 36]
n_jobs = 4


# -- Continuous

cross_fold_ldcrf(mat='../Dataset/NATOPS/NT.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)