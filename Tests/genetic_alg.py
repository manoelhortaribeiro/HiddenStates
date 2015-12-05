import random
import functools
import numpy
import multiprocessing
from multiprocessing.pool import ThreadPool

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


def load_all_folds(folds):

        x, y, x_t, y_t = load_data(folds["Validation"][2], folds["Validation"][3],
                                   folds["Validation"][0], folds["Validation"][1])
        FOLDs["Validation"] = (x, y, x_t, y_t)
        x, y, x_t, y_t = load_data(folds["Test"][2], folds["Test"][3],
                                   folds["Test"][0], folds["Test"][1])
        FOLDs["Test"] = (x, y, x_t, y_t)

# ----------------- Fitness ----------------- #

memory = {}

def test_case(x, y, x_t, y_t, states):
    latent_pbl = GraphLDCRF(n_states_per_label=states, inference_method='dai')
    base_ssvm = NSlackSSVM(latent_pbl, C=1, tol=.01, inactive_threshold=1e-3, batch_size=10, verbose=0, n_jobs=1)
    latent_svm = LatentSSVM(base_ssvm=base_ssvm, latent_iter=5)
    latent_svm.fit(x, y)

    test = latent_svm.score(x_t, y_t)

    return test


def eval_data_set(states, fold, sample):
    x, y, x_t, y_t = fold
    total = 0

    if memory.has_key(tuple(states)):
        result = memory[tuple(states)]
    else:
        for i in range(sample):
            result = test_case(x, y, x_t, y_t, states)
            total += result

        result = total/float(sample)
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
        while (ind[next] <= 1):
            next = random.randrange(0, n_labels)
        ind[next] -= 1

    while sum(ind) < init:
        next = random.randrange(0, n_labels)
        ind[next] += 1

    return ind


def funky_crossover(ind1, ind2):
    init = sum(ind1)
    n_labels = len(ind1)

    ind1, ind2 = tools.cxOnePoint(ind1, ind2)

    # print(ind1[:strip_start] + ind2[strip_start:strip_end] + ind1[strip_end:])
    ind1 = adjust(ind1, init, n_labels)
    ind2 = adjust(ind2, init, n_labels)
    return ind1, ind2


def funky_mutation(ind1):

    times = random.randrange(1, 6)

    lenof = len(ind1)
    indnew = ind1

    for i in range(times):
        points = random.sample(range(lenof), 2)

        if indnew[points[1]] > 1:
            indnew[points[0]] += 1
            indnew[points[1]] -= 1

    return indnew,


def setup(init, t_size, n_labels, p_size):

    toolbox = base.Toolbox()
    initializator = functools.partial(distribute, x=init, n_labels=n_labels)
    pool = ThreadPool(processes=20)

    # creates a fitness that MAXIMIZES the first objective
    creator.create("Fitness", base.Fitness, weights=(1.0,))

    # creates list individual
    creator.create("Individual", list, fitness=creator.Fitness)

    # register everything
    toolbox.register("atrr", initializator)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.atrr, )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", funky_crossover)
    toolbox.register("mutate", funky_mutation)
    toolbox.register("select", tools.selTournament, tournsize=t_size)
    toolbox.register("map", pool.map)
    toolbox.register("evalval", eval_data_set, fold=FOLDs["Validation"], sample=2)

    # initializes population randomly
    pop = toolbox.population(n=p_size)

    # creates stats
    stats = tools.Statistics(key=lambda a: a.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    logbook = tools.Logbook()
    logbook.header = "gen", "avg", "max", "min", "std"

    return toolbox, logbook, stats, pop


# ----------------- Genetic Program ----------------- #

def main(n_labels, folds, init, p_size, CXPB, MUTPB, NGEN, t_size, seed, elite_size, rd=False):

    # seed random generators
    random.seed(seed)
    numpy.random.seed(seed)

    # load folds
    load_all_folds(folds)

    toolbox, logbook, stats, pop = setup(init, t_size, n_labels, p_size)

    # --- First Evaluation
    fitnesses = toolbox.map(toolbox.evalval, pop)

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hall_of_fame_all = []

    for g in range(NGEN):

        hall_of_fame = map(toolbox.clone, tools.selBest(pop, elite_size))

        offspring = toolbox.select(pop, len(pop[:]))

        for i in hall_of_fame:
            hall_of_fame_all.append((g, i, i.fitness.values))
            print ">>>", i, i.fitness.values

        if rd is False:
            offspring = map(toolbox.clone, offspring)

            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)

            # Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)

            # Evaluate the individuals
            fitnesses = toolbox.map(toolbox.evalval, offspring)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit

        if rd is True:
            pop = toolbox.population(n=p_size)

        # The population is entirely replaced by the offspring + the hall of fame
        offspring = tools.selBest(offspring + hall_of_fame, len(pop))
        pop[:] = offspring

        record = stats.compile(pop)
        logbook.record(gen=g, **record)

    arbitrary = divide_hidden_states_arbitrary(init, n_labels)
    best = tools.selBest(pop, 5)
    best.append(arbitrary)

    toolbox.register("evaltest", eval_data_set, fold=FOLDs["Test"], sample=3)

    results = toolbox.map(toolbox.evaltest, best)

    final = zip(best, results)
    print "FINAL (last is arbitrary):", final

    return zip(pop, fitnesses), logbook.stream, hall_of_fame_all, final
