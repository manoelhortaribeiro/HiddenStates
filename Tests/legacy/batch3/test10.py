import Tests.hsparallel_alt as hs
import aux

__author__ = 'Manoel Ribeiro'

tests = [20, 30, 40, 50, 60, 70, 80, 90]
description = "semcad120_7i1s_t20-90"
svmiter = 7
seed = 1
n_jobs = 1
detailed = "Full random init, seed 1, jobs 3, svmiter 7"

n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.cad120()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind="Equal",
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs, measure="stderror", subopt=False)

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)

tests = [20, 30, 40, 50, 60, 70, 80, 90]
description = "semcad120_7i3s_t20-90"
svmiter = 7
seed = 3
n_jobs = 1
detailed = "Full random init, seed 3, jobs 3, svmiter 7"

n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.cad120()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind="Equal",
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs, measure="stderror", subopt=False)

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)