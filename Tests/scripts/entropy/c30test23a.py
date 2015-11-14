import Tests.hsparallel_alt as hs
import aux

__author__ = 'Manoel Ribeiro'

tests = [48,42,36,24,12,6]

description = "ag3f_7i1COSINEsc3_t6-48"
svmiter = 7
seed = 1
n_jobs = 5
detailed = "Full random init, seed 1, jobs 4, svmiter 7"

datapath, n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.armgesturethreefold()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind=0.3,
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs, subopt=False, datapath=datapath,
                     measure="cosine")

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)

description = "ag3f_7i1CORRELATIONsc3_t6-48"

detailed = "Full random init, seed 1, jobs 4, svmiter 7"

datapath, n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.armgesturethreefold()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind=0.3,
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs, subopt=False, datapath=datapath,
                     measure="correlation")

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)

description = "ag3f_7i1EUCLIDIANsc3_t6-48"

detailed = "Full random init, seed 1, jobs 4, svmiter 7"

datapath, n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.armgesturethreefold()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind=0.3,
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs, subopt=False, datapath=datapath,
                     measure="sqeuclidian")

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)

