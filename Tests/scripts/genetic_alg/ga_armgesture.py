import Tests.genetic_alg as ga
import datetime

__author__ = 'Manoel Ribeiro'


n_labels = 6
folds = [1]#, 2, 3, 4, 5]
path = "/home/manoel/Projects/hidden_states_entropy/Dataset/Data/ArmGestureContinuous/"
data = "data"
label = "seqLabels"
train = "Train"
test = "Test"
name = "ArmGestureContinuous"
fold = "Fold"

tests = ga.main(n_labels, folds, path, data, label, train, test, name, fold,
                init=10, p_size=12)

date = datetime.datetime.utcnow().strftime("%d_%m_%y-%H:%M")
description = "GA_armgesture"

project_folder = "/home/manoel/Projects/"
out = "hidden_states_entropy/Dataset/Output/Results/"

f = open(project_folder + out + description + date, "a")

f.write("Number of latent iterations in the SSVM: 10\n")

f.write("Best_Fitness: ")
f.write(str(tests))
f.write("\n")

