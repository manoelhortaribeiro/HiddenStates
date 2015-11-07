import Tests.hs as hs
import datetime

__author__ = 'Manoel Ribeiro'


tests = [50]
n_labels = 10
folds = [1, 2, 3, 4]
path = "/home/manoel/Projects/hidden_states_entropy/Dataset/Data/CAD120/"
data = "data"
label = "seqLabels"
train = "Train"
test = "Test"
name = "CAD120"
fold = "Fold"
date = datetime.datetime.utcnow().strftime("%d_%m_%y-%H:%M")

svmiter = 60

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold, svmiter=svmiter)

description = "CAD120_60iter_50states"


project_folder = "/home/manoel/Projects/"
out = "hidden_states_entropy/Dataset/Output/Results/"

f = open(project_folder + out + description + date, "a")

f.write("Number of latent iterations in the SSVM: " + str(svmiter) + "\n")

f.write("Number of states used in each test: " + str(tests) + "\n")

f.write("Optimal-test (avg, std): ")
f.write(str(opt_tests_avg_std))
f.write("\n")

f.write("Optimal-train (avg, std): ")
f.write(str(opt_trains_avg_std))
f.write("\n")

f.write("Sub-Optimal-test (avg, std): ")
f.write(str(sopt_tests_avg_std))
f.write("\n")

f.write("Sub-Optimal-train (avg, std): ")
f.write(str(sopt_trains_avg_std))
f.write("\n")
