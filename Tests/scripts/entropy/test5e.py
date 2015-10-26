import Tests.hsparallel as hs
import aux

__author__ = 'Manoel Ribeiro'

tests = [20]
svmiter = 100
n_jobs = 1
seed = 1
description = "cad120_100i1s_t20"
detailed = "Full random init, seed 1, jobs 1, svmiter 100"

n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.cad120()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind="Equal",
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs)

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)