import scipy.spatial.distance as distance
from Tests.crossfoldcrf import cross_fold_ldcrf
import functools

__author__ = 'Manoel Ribeiro'

labels = 2
number_folds = 5
states = [2, 4, 6, 8]
n_jobs = 5

datasets_experiment_one = ['0_12345', '01_2345', '012_345']
datasets_experiment_two = ['12_0345', '23_0145', '34_0125', '45_0123']

partial = functools.partial(cross_fold_ldcrf, dist=distance.sqeuclidean, labels=labels,
                            number_folds=number_folds, states=states, n_jobs=n_jobs, c=1)

continuous_experiment_one = map(lambda a: '../Dataset/NATOPS/'+a+'c.mat', datasets_experiment_one)
discrete_experiment_one = map(lambda a: '../Dataset/NATOPS/'+a+'d.mat', datasets_experiment_one)

continuous_experiment_two = map(lambda a: '../Dataset/NATOPS/'+a+'c.mat', datasets_experiment_two)
discrete_experiment_two = map(lambda a: '../Dataset/NATOPS/'+a+'d.mat', datasets_experiment_two)

# -- Continuous
map(partial, continuous_experiment_one)
map(partial, continuous_experiment_two)

# -- Discrete
map(partial, discrete_experiment_one)
map(partial, discrete_experiment_two)

