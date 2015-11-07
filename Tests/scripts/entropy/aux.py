import datetime



def armgesture2():
    n_labels = 6
    folds = [1, 2, 3, 4, 5]
    path = "/home/bruno.teixeira/distance1/hidden_states/Dataset/Data/ArmGestureContinuous/"
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


def write_file(project_folder, out, description, date, svmiter, tests, opt_tests_avg_std,
              opt_trains_avg_std, sopt_tests_avg_std, sopt_trains_avg_std, detailed):
    f = open(project_folder + out + description + date, "a")

    f.write(detailed + "\n")

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
    f.close()
