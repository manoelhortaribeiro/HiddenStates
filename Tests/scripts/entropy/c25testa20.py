import Tests.hsparallel_alt as hs
import aux

__author__ = 'Manoel Ribeiro'

tests = [6, 10, 14, 18, 22, 26, 30]

description = "NEWag_7i1sc25_t6-30"
svmiter = 7
seed = 1
n_jobs = 4
detailed = "Full random init, seed 1, jobs 4, svmiter 7"

n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.armgesture()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind=0.25,
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs)

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)

