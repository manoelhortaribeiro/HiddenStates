import Tests.genetic_alg as ga
import aux

__author__ = 'Manoel Ribeiro'

n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.cad120()

description = "CAD120_GA_seed1"

tests, logbook = ga.main(n_labels, folds, path, data, label, train, test, name,
                         fold, init=6, p_size=4, CXPB=0.5, MUTPB=0.2, NGEN=4, t_size=2, seed=1)

aux.write_file(project_folder, out, description, date, tests, logbook)


