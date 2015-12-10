import aux
import Tests.hsparallel_alt as hs

__author__ = 'manoel'

n_labels, folds, date, out, datapath = aux.armgesture()

description = "ntcontinuoustest_"

test_our, train_our = hs.do_test(6, 6, "sqeuclidian", datapath, 3, folds, 1)

test_arb, train_arb = hs.do_test(6, 6, "arbitrary", datapath, 3, folds, 1)

aux.write_file(out, description, date, test_our, train_our, test_arb, train_arb)
