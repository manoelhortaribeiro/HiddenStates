import scipy.spatial.distance as distance

from Tests.crossfoldcrf import cross_fold_ldcrf

__author__ = 'Manoel Ribeiro'

labels = 2
number_folds = 5
states = [2, 4, 6, 8]
n_jobs = 4


# -- Continuous

cross_fold_ldcrf(mat='../Dataset/NATOPS/0_12345c.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=0.25)

cross_fold_ldcrf(mat='../Dataset/NATOPS/01_2345c.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=0.25)

cross_fold_ldcrf(mat='../Dataset/NATOPS/012_345c.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=0.25)

# -- Discrete

cross_fold_ldcrf(mat='../Dataset/NATOPS/0_12345d.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=0.25)

cross_fold_ldcrf(mat='../Dataset/NATOPS/01_2345d.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=0.25)

cross_fold_ldcrf(mat='../Dataset/NATOPS/012_345d.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=0.25)

