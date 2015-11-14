import Tests.hsparallel_alt as hs
import aux

__author__ = 'Manoel Ribeiro'

tests = [48,42,36,24,12,6]

description = "ag3f_7i1sc3_t6-48"
svmiter = 7
seed = 1
n_jobs = 3
detailed = "Full random init, seed 1, jobs 4, svmiter 7"

datapath, n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.armgesturethreefold()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind=0.3,
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs, subopt=True, opt=False, datapath=datapath,
                     measure="cosine")

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)

description = "ag3f_7i2sc3_t6-48"
seed = 2

detailed = "Full random init, seed 2, jobs 4, svmiter 7"

datapath, n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.armgesturethreefold()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind=0.3,
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs, subopt=True, opt=False, datapath=datapath,
                     measure="correlation")

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)

description = "ag3f_7i3sc3_t6-48"
seed = 3

detailed = "Full random init, seed 3, jobs 4, svmiter 7"

datapath, n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.armgesturethreefold()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind=0.3,
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs, subopt=True, opt=False, datapath=datapath,
                     measure="sqeuclidian")

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)

