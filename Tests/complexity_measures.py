import scipy.spatial.distance as distance
from Util.data_parser import load_data
from measures import *
import functools

__author__ = 'Manoel Ribeiro'


def complexity_measure(mat, dist):
    x, y = load_data(mat)
    prop = calculate_dist(y, x, dist)

    f = open("../Output/Results/complexity.txt", "a+")

    f.write('Mat:' + mat + ',' + str(prop) + '\n')



# ArmGesture

datasets_experiment_one = ['0_12345', '01_2345', '012_345']
datasets_experiment_two = ['12_0345', '23_0145', '34_0125', '45_0123']

list_databases = []

list_databases += map(lambda a: '../Dataset/ArmGesture/'+a+'c.mat', datasets_experiment_one)
list_databases += map(lambda a: '../Dataset/ArmGesture/'+a+'d.mat', datasets_experiment_one)
list_databases += map(lambda a: '../Dataset/ArmGesture/'+a+'c.mat', datasets_experiment_two)
list_databases += map(lambda a: '../Dataset/ArmGesture/'+a+'d.mat', datasets_experiment_two)
list_databases += map(lambda a: '../Dataset/NATOPS/'+a+'c.mat', datasets_experiment_one)
list_databases += map(lambda a: '../Dataset/NATOPS/'+a+'d.mat', datasets_experiment_one)
list_databases += map(lambda a: '../Dataset/NATOPS/'+a+'c.mat', datasets_experiment_two)
list_databases += map(lambda a: '../Dataset/NATOPS/'+a+'d.mat', datasets_experiment_two)

partial = functools.partial(complexity_measure, dist=distance.euclidean)

map(partial, list_databases)
