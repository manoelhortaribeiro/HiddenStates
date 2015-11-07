import Tests.hs as hs
import aux

__author__ = 'Manoel Ribeiro'


n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.cad120()


tests = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
svmiter = 10
seed = 1
description = "CAD120c30"
n_jobs = 8

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold,
                     kind="Equal", svmiter=svmiter, seed=seed, n_jobs=n_jobs)


aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std)
