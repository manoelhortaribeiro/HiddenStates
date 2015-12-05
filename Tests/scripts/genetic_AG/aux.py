import datetime

def armgesture():
    n_labels = 6
    path = "/home/bruno.teixeira/ga2/hidden_states/Dataset/Data/ArmGesture/GA_Discrete"
    folds = {}
    folds["Validation"] = ( path + "/Validation/dataValidationTest.csv", path + "/Validation/seqLabelsValidationTest.csv",
                           path + "/Validation/dataValidationTrain.csv", path + "/Validation/seqLabelsValidationTrain.csv")
    folds["Test"] = ( path + "/Test/dataTestTest.csv", path + "/Test/seqLabelsTestTest.csv",
                      path + "/Test/dataTestTrain.csv", path + "/Test/seqLabelsTestTrain.csv")

    date = datetime.datetime.utcnow().strftime("%d_%m_%y-%H:%M")
    out = "/home/bruno.teixeira/ga2/hidden_states/Dataset/Output/Results/"
    return n_labels, folds, date, out



def armgesture2():
    n_labels = 6
    path = "/home/manoel/Projects/hidden_states_entropy/Dataset/Data/ArmGesture/GA_Discrete"
    folds = {}
    folds["Validation"] = ( path + "/Validation/dataValidationTest.csv", path + "/Validation/seqLabelsValidationTest.csv",
                           path + "/Validation/dataValidationTrain.csv", path + "/Validation/seqLabelsValidationTrain.csv")
    folds["Test"] = ( path + "/Test/dataTestTest.csv", path + "/Test/seqLabelsTestTest.csv",
                      path + "/Test/dataTestTrain.csv", path + "/Test/seqLabelsTestTrain.csv")

    date = datetime.datetime.utcnow().strftime("%d_%m_%y-%H:%M")
    out = "/home/manoel/Projects/hidden_states_entropy/Dataset/Output/Results/"

    return n_labels, folds, date, out


def write_file(out, description, date, tests, logbook, best, svmiter, arbitrary):

    f = open(out + description + date, "a")

    f.write(logbook)
    f.write("\nNumber of latent iterations in the SSVM: " + str(svmiter))
    f.write("\n(Ind,Fitness): ")
    f.write(str(tests))

    f.write("\nBest Individuals: ")
    f.write("\n|- Gen -|-------------- States ------------|--------Fitness-------|")

    for b in best:
        f.write("\n|   " + str(b[0]) + "        " + str(b[1]) + "         " + str(b[2])[1:9])

    f.write("\nVALIDATION\nARBITRARY, OURS: ")
    f.write(str(arbitrary))
    f.close()

