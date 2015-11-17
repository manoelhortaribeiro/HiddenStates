import Tests.hsparallel_alt as hs
import aux

__author__ = 'Manoel Ribeiro'

tests = [35, 40, 45, 50]

description = "NEWag_7i2s_t35-50"
svmiter = 7
seed = 3
n_jobs = 3
detailed = "Full random init, seed 3, jobs 4, svmiter 7"

n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.armgesture()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind=1,
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs)

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)

