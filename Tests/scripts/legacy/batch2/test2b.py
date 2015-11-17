import Tests.hsparallel as hs
import aux

__author__ = 'Manoel Ribeiro'

tests = [20, 30, 40, 50]
description = "cad120p_7i8s_t20-50"
svmiter = 7
seed = 8
n_jobs = 3
detailed = "Full random init, seed 8, jobs 3, svmiter 7"

n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.cad120()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind="Equal",
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs)

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)