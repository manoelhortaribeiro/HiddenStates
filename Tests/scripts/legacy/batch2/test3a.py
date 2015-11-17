import Tests.hs as hs
import aux

__author__ = 'Manoel Ribeiro'

tests = [30]
svmiter = 75
n_jobs = 8
seed = 1
description = "cad120_75i1s_t30"
detailed = "Full random init, seed 1, jobs 8, svmiter 75"

n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.cad120()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind="Equal",
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs)

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)