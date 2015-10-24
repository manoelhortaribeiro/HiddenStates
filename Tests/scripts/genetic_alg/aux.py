import datetime

def cad120():
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
    project_folder = "/home/manoel/Projects/"
    out = "hidden_states_entropy/Dataset/Output/Results/"

    return n_labels, folds, path, data, label, train, test, name, fold, date, project_folder, out


def write_file(project_folder, out, description, date, tests, logbook, best, svmiter):

    f = open(project_folder + out + description + date, "a")

    f.write(logbook)
    f.write("\nNumber of latent iterations in the SSVM: " + str(svmiter))
    f.write("\n(Ind,Fitness): ")
    f.write(str(tests))

    f.write("\nBest Individuals: ")
    f.write("\n|- Gen -|-------------------- States ---------------|--------Fitness-------|- Fold -|")

    for b in best:
        f.write("\n|   " + str(b[0]) + "           " + str(b[1]) + "           " + str(b[2])[1:9] + "              " + str(b[3]))
    f.close()

