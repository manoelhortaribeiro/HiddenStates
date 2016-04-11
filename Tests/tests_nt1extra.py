import scipy.spatial.distance as distance

from Tests.crossfoldcrf import cross_fold_ldcrf

__author__ = 'Manoel Ribeiro'

labels = 2
number_folds = 5
states = [2, 4, 6, 8]
n_jobs = 5

# -- Discrete

cross_fold_ldcrf(mat='../Dataset/NATOPS/12_0345d.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

cross_fold_ldcrf(mat='../Dataset/NATOPS/23_0145d.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

cross_fold_ldcrf(mat='../Dataset/NATOPS/34_0125d.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

cross_fold_ldcrf(mat='../Dataset/NATOPS/45_0123d.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

exit()
