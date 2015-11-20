import datetime

def armgesture():
    n_labels = 6
    folds = [1, 2, 3]
    path = "/home/bruno.teixeira/distance6/hidden_states/Dataset/Data/ArmGestureContinuous3FoldHALF/"
    data = "data"
    label = "seqLabels"
    train = "Train"
    test = "Test"
    name = "ArmGestureContinuous"
    fold = "FoldHALF"
    date = datetime.datetime.utcnow().strftime("%d_%m_%y-%H:%M")
    project_folder = "/home/bruno.teixeira/distance6/"
    out = "hidden_states/Dataset/Output/Results/"
    datapath = path + "/ArmGestureDiscrete.mat"

    return datapath, n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out



def armgesture2():
    n_labels = 6
    folds = [1, 2, 3]
    path = "/home/bruno.teixeira/distance6/hidden_states/Dataset/Data/ArmGestureContinuous/"
    data = "data"
    label = "seqLabels"
    train = "Train"
    test = "Test"
    name = "ArmGestureContinuous"
    fold = "Fold"
    date = datetime.datetime.utcnow().strftime("%d_%m_%y-%H:%M")
    project_folder = "/home/bruno.teixeira/distance1/"
    out = "hidden_states/Dataset/Output/Results/"
    datapath = path + "/ArmGestureDiscrete.mat"

    return datapath, n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out



def write_file(project_folder, out, description, date, tests, logbook, best, svmiter, arbitrary):

    f = open(project_folder + out + description + date, "a")

    f.write(logbook)
    f.write("\nNumber of latent iterations in the SSVM: " + str(svmiter))
    f.write("\n(Ind,Fitness): ")
    f.write(str(tests))

    f.write("\nBest Individuals: ")
    f.write("\n|- Gen -|-------------- States ------------|--------Fitness-------|")

    for b in best:
        f.write("\n|   " + str(b[0]) + "        " + str(b[1]) + "         " + str(b[2])[1:9])

    f.write("\nARBITRARY: ")
    f.write(str(arbitrary))
    f.close()

