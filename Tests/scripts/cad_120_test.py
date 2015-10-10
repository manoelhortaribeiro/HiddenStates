import Tests.hidden_states as hs
import datetime

__author__ = 'Manoel Ribeiro'


tests = [12, 14, 16, 18, 20, 24, 28, 32, 36]
n_labels = 6
folds = [1, 2, 3, 4, 5]
path = "/home/manoel/Projects/hidden_states_entropy/Dataset/Data/ArmGestureContinuous/"
data = "data"
label = "seqLabels"
train = "Train"
test = "Test"
name = "ArmGestureContinuous"
fold = "Fold"

opt_tests_avg_std, opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std = \
    [1, 2, 3], [1, 2, 3], [1, 3, 4], [1, 3, 4]

    #hs.eval_data_set(tests, n_labels, folds, path, data, label, train, test, name, fold)

date = datetime.datetime.utcnow().strftime("%d_%m_%y-%H:%M")
description = "ArmGestureContinuous"


project_folder = "/home/manoelribeiro/"
out = "hidden_states_entropy/Dataset/Output/Results/"

f = open(project_folder + out + description + date, "a")

f.write("Number of latent iterations in the SSVM: 10\n")

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
