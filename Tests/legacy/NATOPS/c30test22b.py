import Tests.hsparallel_alt as hs
import aux

__author__ = 'Manoel Ribeiro'

tests = [30,24,18,12,6]

description = "NT_7i2COSINEsc3_t6-30"
svmiter = 7
seed = 2
n_jobs = 3
detailed = "Full random init, seed2, jobs 4, svmiter 7"

datapath, n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.NATOPS()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind=0.3,
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs, subopt=True, datapath=datapath,
                     measure="cosine")

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)

description = "NT_7i2CORRELATIONsc3_t6-30"

detailed = "Full random init, seed2, jobs 4, svmiter 7"

datapath, n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.NATOPS()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind=0.3,
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs, subopt=False, datapath=datapath,
                     measure="correlation")

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)

description = "NT_7i2EUCLIDIANsc3_t6-30"

detailed = "Full random init, seed2, jobs 4, svmiter 7"

datapath, n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out = aux.NATOPS()

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, kind=0.3,
                     svmiter=svmiter, seed=seed, n_jobs=n_jobs, subopt=False, datapath=datapath,
                     measure="sqeuclidian")

aux.write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
               opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed)

