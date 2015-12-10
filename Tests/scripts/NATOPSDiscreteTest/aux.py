import datetime


def armgesture2():
    n_labels = 6
    path = "/home/manoel/Projects/hidden_states_entropy/Dataset/Data/NATOPS/Discrete"

    folds = (path + "1/dataNATOPSDiscrete1.csv", path + "1/seqLabelsNATOPSDiscrete1.csv",
                     path + "2/dataNATOPSDiscrete2.csv", path + "2/seqLabelsNATOPSDiscrete2.csv")

    date = datetime.datetime.utcnow().strftime("%d_%m_%y-%H:%M")
    out = "/home/manoel/Projects/hidden_states_entropy/Dataset/Output/Results/"
    datapath = "/home/manoel/Projects/hidden_states_entropy/Dataset/Data/NATOPS/Discrete3/" + "NATOPSDiscrete3.mat"

    return n_labels, folds, date, out, datapath

def armgesture():
    n_labels = 6
    path = "/home/bruno.teixeira/nttest/hidden_states/Dataset/Data/NATOPS/Discrete"

    folds = (path + "1/dataNATOPSDiscrete1.csv", path + "1/seqLabelsNATOPSDiscrete1.csv",
                     path + "2/dataNATOPSDiscrete2.csv", path + "2/seqLabelsNATOPSDiscrete2.csv")

    date = datetime.datetime.utcnow().strftime("%d_%m_%y-%H:%M")
    out = "/home/bruno.teixeira/nttest/hidden_states/Dataset/Output/Results/"
    datapath = "/home/bruno.teixeira/nttest/hidden_states/Dataset/Data/NATOPS/Discrete3/" + "NATOPSDiscrete3.mat"

    return n_labels, folds, date, out, datapath

def write_file(out, description, date, testour, trainour, testarb, trainarb):

    f = open(out + description + date, "a")

    f.write("Ours (test,train):" + str(testour) + str(trainour) + "\n")

    f.write("Arb (test,train):" + str(testarb) + str(trainarb) + "\n")

    f.close()