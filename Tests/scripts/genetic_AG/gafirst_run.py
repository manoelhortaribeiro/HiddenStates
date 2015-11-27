import Tests.genetic_alg as ga
import aux

__author__ = 'Manoel Ribeiro'


def run_it(seed, description):

    data_path, n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.armgesture()

    init = 30
    p_size = 20
    CXPB = 0.6
    MUTPB = 0.6
    NGEN = 100
    t_size = 2
    svm = 5
    elite_size = 1

    tests, logbook, best, arbitrary = ga.main(n_labels, folds, path, data, label, train, test, name, fold,
                                   init=init, p_size=p_size, CXPB=CXPB, MUTPB=MUTPB, NGEN=NGEN,
                                   t_size=t_size, seed=seed, elite_size=elite_size)

    aux.write_file(project_folder, out, description, date, tests, logbook, best, svm, arbitrary)

desc = "init_36_psize_20_CXPB_6_MUTPB_6_NGEN_20_tsize_2_elite_1"
run_it(seed=1, description="GA1"+desc)
run_it(seed=2, description="GA2"+desc)
run_it(seed=3, description="GA3"+desc)
run_it(seed=4, description="GA4"+desc)
run_it(seed=5, description="GA5"+desc)







