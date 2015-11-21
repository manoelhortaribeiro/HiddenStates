import random
import functools
import numpy
import multiprocessing
# Deap Imports
from deap import base
from deap import creator
from deap import tools
# Deap Imports
from pystruct.learners import NSlackSSVM, LatentSSVM
# Internal Imports
from Util.data_parser import load_data
from Models.GraphLDCRF import GraphLDCRF
from Tests.measures import divide_hidden_states_arbitrary

__author__ = 'Manoel Ribeiro'
# ----------------- I/O ----------------- #

FOLDs = {}


def load_all_folds(path, data, label, train, test, name, fold, folds):
    tmp = ["TRAIN", "TEST", "VALIDATION"]

    for idx, i in enumerate(folds):
        # test
        dte = path + data + test + name + fold + str(i) + ".csv"
        sqte = path + label + test + name + fold + str(i) + ".csv"

        # train
        dtr = path + data + train + name + fold + str(i) + ".csv"
        sqtr = path + label + train + name + fold + str(i) + ".csv"

        x, y, x_t, y_t = load_data(dtr, sqtr, dte, sqte)
        FOLDs[tmp[idx]] = (x, y, x_t, y_t)


# ----------------- Fitness ----------------- #

memory = {}


def test_case(x, y, x_t, y_t, states):
    latent_pbl = GraphLDCRF(n_states_per_label=states, inference_method='dai')
    base_ssvm = NSlackSSVM(latent_pbl, C=1, tol=.01, inactive_threshold=1e-3, batch_size=10, verbose=0, n_jobs=1)
    latent_svm = LatentSSVM(base_ssvm=base_ssvm, latent_iter=5)
    latent_svm.fit(x, y)

    test = latent_svm.score(x_t, y_t)

    return test


def eval_data_set(states, foldtrain, foldtest):
    garbage1, garbage2, x, y = foldtrain
    garbage1, garbage2, x_t, y_t = foldtest

    if memory.has_key(tuple(states)):
        result = memory[tuple(states)]
    else:
        result = test_case(x, y, x_t, y_t, states)
        memory[tuple(states)] = result

    return result,


# ----------------- Helpers ----------------- #


def distribute(x, n_labels):
    balls_left = x

    list = []

    for i in range(n_labels):
        list.append(1)

    balls_left = balls_left - n_labels

    while (balls_left != 0):
        next = random.randrange(n_labels)
        list[next] += 1
        balls_left -= 1

    return list


def adjust(ind, init, n_labels):
    if sum(ind) is init:
        return ind

    while sum(ind) > init:
        next = random.randrange(0, n_labels)
        while (next <= 1):
            next = random.randrange(0, n_labels)
        ind[next] -= 1

    while sum(ind) < init:
        next = random.randrange(0, n_labels)
        ind[next] += 1

    return ind


def funky_crossover(ind1, ind2):
    init = sum(ind1)
    n_labels = len(ind1)

    ind1, ind2 = tools.cxTwoPoint(ind1, ind2)

    # print(ind1[:strip_start] + ind2[strip_start:strip_end] + ind1[strip_end:])
    ind1 = adjust(ind1, init, n_labels)
    ind2 = adjust(ind2, init, n_labels)

    return ind1, ind2


def funky_mutation(ind1):
    lenof = len(ind1)
    initsum = numpy.array(ind1).sum()
    points = random.sample(range(lenof), 2)

    indnew = ind1
    if indnew[points[1]] > 1:
        indnew[points[0]] += 1
        indnew[points[1]] -= 1

    if numpy.array(indnew).sum() != initsum:
        print "ERROR"
        quit()
    return indnew,


def setup(init, t_size, n_labels):
    toolbox = base.Toolbox()

    # initialize high order functions
    initializator = functools.partial(distribute, x=init, n_labels=n_labels)

    # register everything
    toolbox.register("atrr", initializator)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.atrr, )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", funky_crossover)

    toolbox.register("mutate", funky_mutation)

    toolbox.register("select", tools.selTournament, tournsize=t_size)

    pool = multiprocessing.Pool(processes=10)
    toolbox.register("map", map)

    pop = toolbox.population(n=4)

    toolbox.register("evaluate", eval_data_set, foldtrain=FOLDs["TRAIN"], foldtest=FOLDs["TEST"])

    # creates stats
    stats = tools.Statistics(key=lambda a: a.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    logbook = tools.Logbook()
    logbook.header = "gen", "avg", "max", "min", "std"

    return toolbox, logbook, stats


# ----------------- Genetic Program ----------------- #

def main(n_labels, folds, path, data, label, train, test, name, fold, init, p_size=3, CXPB=0.8,
         MUTPB=0.2, NGEN=4, t_size=2, seed=1, elite_size=1):
    # seed random generators
    random.seed(seed)
    numpy.random.seed(seed)

    # load folds
    load_all_folds(path, data, label, train, test, name, fold, folds)

    # creates a fitness that MAXIMIZES the first objective
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))

    # creates list individual
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox, logbook, stats = setup(init, t_size, n_labels)

    pop = toolbox.population(n=p_size)


    # evaluate the entire population
    fitnesses = toolbox.map(toolbox.evaluate, pop)

    hall_of_fame_all = []

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):

        print g, "/", NGEN, "len:", len(pop[:])

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop[:]))

        hall_of_fame = tools.selBest(offspring, elite_size)

        for i in hall_of_fame:
            hall_of_fame_all.append((g, i, i.fitness.values))
            print i, i.fitness.values

        if g is NGEN - 1:
            break

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

        offspring = tools.selBest(offspring + hall_of_fame, len(pop))

        # Delete all fitness values
        for i in offspring:
            del i.fitness.values

        for i in hall_of_fame:
            del i.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        # The population is entirely replaced by the offspring
        pop[:] = offspring

        record = stats.compile(pop)
        logbook.record(gen=g, **record)

    arbitrary = divide_hidden_states_arbitrary(init, n_labels)

    best = tools.selBest(pop, 1)
    arbitrary_acc, ours_acc = eval_data_set(arbitrary, FOLDs["TRAIN"], FOLDs["VALIDATION"]), \
                              eval_data_set(best[0], FOLDs["TRAIN"], FOLDs["VALIDATION"])

    return zip(pop, fitnesses), logbook.stream, hall_of_fame_all, (arbitrary_acc, ours_acc)
