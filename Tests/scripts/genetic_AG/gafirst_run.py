import Tests.genetic_alg as ga
import aux

__author__ = 'Manoel Ribeiro'

data_path, n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.armgesture()

description = "run0_CAD120_GA_seed1"

init = 6
p_size = 15
CXPB = 0.6
MUTPB = 0.2
NGEN = 10
t_size = 3
seed = 1
svm = 5
elite_size = 1

tests, logbook, best = ga.main(n_labels, folds, path, data, label, train, test, name, fold,
                               svm=svm, init=init, p_size=p_size, CXPB=CXPB, MUTPB=MUTPB, NGEN=NGEN,
                               t_size=t_size, seed=seed, elite_size=elite_size)

aux.write_file(project_folder, out, description, date, tests, logbook, best, svm)


