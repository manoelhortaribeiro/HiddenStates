from val import *

dataset = "NATOPS"

val_datatrain = dataset + "/Continuous1/dataTrain" + dataset + "ContinuousFold1"
val_datatest = dataset + "/Continuous1/dataTest" + dataset + "ContinuousFold1"
val_seqtrain = dataset + "/Continuous1/seqLabelsTrain" + dataset + "ContinuousFold1"
val_seqtest = dataset + "/Continuous1/seqLabelsTest" + dataset + "ContinuousFold1"
tst_datatrain = dataset + "/Continuous/dataTrain.csv"
tst_datatest = dataset + "/Continuous/dataTest.csv"
tst_seqtrain = dataset + "/Continuous/seqLabelsTrain.csv"
tst_seqtest = dataset + "/Continuous/seqLabelsTest.csv"

x, x_t, y, y_t, t_x, t_x_t, t_y, t_y_t = load(val_datatrain, val_datatest, val_seqtrain,
                                              val_seqtest, tst_datatrain, tst_datatest,
                                              tst_seqtrain, tst_seqtest)

print "GREEDY:"

greedy_states, g_test = greedy(x, y, x_t, y_t, add=1, rangeof=6)

print "ARBITRARY:"
arbitrary_states, a_test = 0, 0 # arbitrary(t_x, t_y, t_x_t, t_y_t, add=1, rangeof=6)

r_arb, r_gre = test(greedy_states, arbitrary_states, x, y, x_t, y_t, numberseeds=3)

print "Results:\n", "arb:", r_arb, "gre:", r_gre

path = os.environ['PYTHONPATH'].split(os.pathsep)[0]
f = open(path + "/Dataset/Output/Results/NTCgreedy", "w")
f.write("arb: " + str(r_arb))
f.write("\ngre: " + str(r_gre))
f.close()
