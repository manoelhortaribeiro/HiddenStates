import random
import functools
import multiprocessing

from deap import base
from deap import creator
from deap import tools
import numpy


from pystruct.learners import NSlackSSVM, LatentSSVM

# Internal Imports
from Util.data_parser import load_data
from Models.GraphLDCRF import GraphLDCRF

__author__ = 'Manoel Ribeiro'

ALLFOLDS = {}


def test_case(svm, x, y, x_t, y_t, states):


    # TEST #
    latent_pbl = GraphLDCRF(n_states_per_label=states, inference_method='dai')
    base_ssvm = NSlackSSVM(latent_pbl, C=1, tol=.01, inactive_threshold=1e-3, batch_size=10, verbose=0, n_jobs=1)
    latent_svm = LatentSSVM(base_ssvm=base_ssvm, latent_iter=svm)
    latent_svm.fit(x, y)

    test = latent_svm.score(x_t, y_t)

    return test


# does all the folds in a data-set
def eval_data_set(svm, i, states):

    x, y, x_t, y_t = ALLFOLDS[i]

    result = test_case(svm, x, y, x_t, y_t, states)

    return result,


def random_thingy(x):
    return random.randrange(1, x)


def load_all_folds(path, data, label, train, test, name, fold, folds):

    for i in folds:
        # test
        dte = path + data + test + name + fold + str(i) + ".csv"
        sqte = path + label + test + name + fold + str(i) + ".csv"

        # train
        dtr = path + data + train + name + fold + str(i) + ".csv"
        sqtr = path + label + train + name + fold + str(i) + ".csv"

        x, y, x_t, y_t = load_data(dtr, sqtr, dte, sqte)
        ALLFOLDS[i] = (x, y, x_t, y_t)


def main(n_labels, folds, path, data, label, train, test, name, fold, init, p_size=10, CXPB=0.5,
         MUTPB=0.2, NGEN=4, svm=7, t_size=2, seed=1):

    random.seed(seed)
    numpy.random.seed(seed)
    load_all_folds(path, data, label, train, test, name, fold, folds)

    print "FOLDS LOADED"
    ind_size = n_labels

    # creates a fitness that minimizes the first objective
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # creates list individual
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # initialize high order functions
    initializator = functools.partial(random_thingy, init)

    toolbox = base.Toolbox()

    # Chooses a fold to evaluate with
    this_fold = random.choice(folds)
    evaluate = functools.partial(eval_data_set, svm, this_fold)
    toolbox.register("evaluate", evaluate)

    # register everything
    toolbox.register("attr_float",  initializator)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=init, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=t_size)
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # creates stats
    stats = tools.Statistics(key=lambda a: a.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    logbook = tools.Logbook()
    logbook.header = "gen", "avg", "max", "min", "std"
    # define constants in the GA

    pop = toolbox.population(n=p_size)

    # evaluate the entire population
    fitnesses = toolbox.map(toolbox.evaluate, pop)

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):

        print g, "/", NGEN

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = toolbox.map(toolbox.clone, offspring)

        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on the offspring
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Chooses a fold to evaluate with
        this_fold = random.choice(folds)
        evaluate = functools.partial(eval_data_set, svm, this_fold)
        toolbox.register("evaluate", evaluate)

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        record = stats.compile(pop)
        logbook.record(gen=g, **record)

    return zip(pop, fitnesses), logbook.stream

