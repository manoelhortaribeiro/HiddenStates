import scipy.spatial.distance as distance

from Tests.crossfoldcrf import cross_fold_ldcrf

__author__ = 'Manoel Ribeiro'

labels = 2
number_folds = 5
states = [2, 4, 6, 8, 10, 12, 14]
n_jobs = 5


# -- Continuous

cross_fold_ldcrf(mat='../Dataset/ArmGesture/12_0345c.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

cross_fold_ldcrf(mat='../Dataset/ArmGesture/23_0145c.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

cross_fold_ldcrf(mat='../Dataset/ArmGesture/34_0125c.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

cross_fold_ldcrf(mat='../Dataset/ArmGesture/45_0123c.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

# -- Discrete

cross_fold_ldcrf(mat='../Dataset/ArmGesture/12_0345d.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

cross_fold_ldcrf(mat='../Dataset/ArmGesture/23_0145d.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

cross_fold_ldcrf(mat='../Dataset/ArmGesture/34_0125d.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

cross_fold_ldcrf(mat='../Dataset/ArmGesture/45_0123d.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

exit()

# -- Continuous

cross_fold_ldcrf(mat='../Dataset/ArmGesture/0_12345c.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

cross_fold_ldcrf(mat='../Dataset/ArmGesture/01_2345c.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

cross_fold_ldcrf(mat='../Dataset/ArmGesture/012_345c.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

# -- Discrete

cross_fold_ldcrf(mat='../Dataset/ArmGesture/0_12345d.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

cross_fold_ldcrf(mat='../Dataset/ArmGesture/01_2345d.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

cross_fold_ldcrf(mat='../Dataset/ArmGesture/012_345d.mat', dist=distance.sqeuclidean,
                 labels=labels, number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)