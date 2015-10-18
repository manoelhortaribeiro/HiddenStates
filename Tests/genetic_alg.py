import random
import functools
import numpy as np

from deap import base
from deap import creator
from deap import tools

from sklearn import metrics
from sklearn.metrics import confusion_matrix

from pystruct.learners import NSlackSSVM, LatentSSVM

# Internal Imports
import Util.pyeeg as pyeeg  # Contains the sample entropy calculation
from Util.data_parser import load_data
from Models.GraphLDCRF import GraphLDCRF

__author__ = 'Manoel Ribeiro'


def test_case(x, y, x_t, y_t, states):


    # TEST #
    latent_pbl = GraphLDCRF(n_states_per_label=states, inference_method='dai')
    base_ssvm = NSlackSSVM(latent_pbl, C=1, tol=.01, inactive_threshold=1e-3, batch_size=10, verbose=0, n_jobs=6)
    latent_svm = LatentSSVM(base_ssvm=base_ssvm, latent_iter=10)
    latent_svm.fit(x, y)

    test = latent_svm.score(x_t, y_t)

    return test


# does all the folds in a data-set
def eval_data_set(folds, path, data, label, train, test, name, fold, states):

    tests = []

    for i in folds:
        # test
        dte = path + data + test + name + fold + str(i) + ".csv"
        sqte = path + label + test + name + fold + str(i) + ".csv"

        # train
        dtr = path + data + train + name + fold + str(i) + ".csv"
        sqtr = path + label + train + name + fold + str(i) + ".csv"

        x, y, x_t, y_t = load_data(dtr, sqtr, dte, sqte)
        test = random.random() #test_case(x, y, x_t, y_t, states)
        tests.append(test)

    tests = 10#np.array(tests)

    # Optimal Data

    tests_avg = 12#tests.mean(0)

    return tests_avg,


def random_thingy(x):
    return random.randrange(1, x)


def main(n_labels, folds, path, data, label, train, test, name,
         fold, init, p_size=10):

    ind_size = n_labels

    # creates a fitness that minimizes the first objective
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # creates list individual
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # initialize high order functions
    initializator = functools.partial(random_thingy, init)

    evaluate = functools.partial(eval_data_set, folds, path, data,label, train, test, name, fold)

    # register everything
    toolbox = base.Toolbox()
    toolbox.register("attr_float",  initializator)
    toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attr_float, n=ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=init, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # define constants in the GA
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40
    pop = toolbox.population(n=p_size)

    # evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    print pop