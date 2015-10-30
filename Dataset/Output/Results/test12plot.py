import matplotlib.pyplot as plt
import matplotlib
from matplotlib.legend_handler import HandlerLine2D
from math import sqrt
import numpy
from numpy import array
SPINE_COLOR = 'gray'


states = [10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90]

optimal_test = []
optimal_test.append(array([ 0.83802142,  0.83521215,  0.8440584,  0.83870311,  0.83023831,
                            0.83793004,  0.81885452,  0.81194642,  0.80685894,
                            0.81593763,  0.83406192,  0.82664688,  0.82547298]))  # seed 1


optimal_test.append(array([ 0.82789693,  0.83141112,  0.8333711,  0.83149333,  0.83276599,
                            0.82274169,  0.8369527 ,  0.82068679,  0.8194296,
                            0.81661298,  0.82526975,  0.79316411,  0.80680319 ]))  # seed 5

optimal_test.append(array([ 0.83121954,  0.83578447,  0.83734432,  0.83842082,  0.8368666,
                            0.82741987,  0.82283307,  0.81029974,  0.80764082,
                            0.81048607,  0.82168738,  0.81766341,  0.82478279]))  # seed 6

#optimal_test.append(array([ 0.84124031,  0.82379561,  0.82451604,  0.79862562]))
#optimal_test.append(array([ 0.82849283,  0.83073825,  0.81642735,  0.81071752]))


optimal_stdev = []
optimal_stdev.append(array([ 0.0183667 ,  0.02200394,  0.01865816,  0.03398549]))
optimal_stdev.append(array([ 0.02869886,  0.02788698,  0.01918655,  0.02073465]))
optimal_stdev.append(array([ 0.02219525,  0.02096007,  0.02910535,  0.03187896]))
optimal_stdev.append(array([ 0.01791165,  0.02475836,  0.01217445,  0.03151903]))
optimal_stdev.append(array([ 0.01713637,  0.03490643,  0.01184646,  0.01620074]))
optimal_stdev.append(array([ 0.01617888,  0.01455587,  0.02293893,  0.01934039]))


suboptimal_test = []
suboptimal_test.append(array([ 0.83620855,  0.82169266,  0.8181644 ,  0.82304228,  0.81816541,
                               0.81553116,  0.8137963 ,  0.79275656,  0.83658144,
                               0.80710911,  0.82283271,  0.80528918,  0.81804971])) # seed 1

suboptimal_test.append(array([ 0.82674281,  0.81380053,  0.82246079,  0.81220894,  0.80981426,
                               0.81069137,  0.81509258,  0.8209821 ,  0.81914066,
                               0.80238884,  0.82187724,  0.81068146,  0.83257833])) # seed 5

suboptimal_test.append(array([ 0.8364845 ,  0.83434312,  0.83005797,  0.81448261,  0.81875184,
                               0.81303174,  0.80454541,  0.81603647,  0.8276933,
                               0.81390213,  0.82799464,  0.81428004,  0.80570447])) # seed 6


suboptimal_stdev = []
suboptimal_stdev.append(array([ 0.03703645,  0.01834699,  0.03145797,  0.03302787]))
suboptimal_stdev.append(array([ 0.00967098,  0.03265915,  0.02777041,  0.01496227]))
suboptimal_stdev.append(array([ 0.01333193,  0.01290421,  0.02864108,  0.02791686]))
suboptimal_stdev.append(array([ 0.01080623,  0.02049345,  0.01732611,  0.03166045]))
suboptimal_stdev.append(array([ 0.02654663,  0.02703446,  0.02688205,  0.0251933 ]))
suboptimal_stdev.append(array([ 0.01849264,  0.02633186,  0.02967562,  0.03666131]))




optimal_test = array(optimal_test).mean(0)
suboptimal_test = array(suboptimal_test).mean(0)
optimal_stdev = numpy.sqrt(array(suboptimal_stdev).__pow__(2).sum(0))
suboptimal_stdev = numpy.sqrt(array(suboptimal_stdev).__pow__(2).mean(0))
print optimal_stdev

plt.xlabel("Number of Hidden States")
plt.ylabel("Accuracy")

lineopt, = plt.plot(states, optimal_test, "g-", label="Optimal")
#plt.errorbar(states, optimal_test, optimal_stdev, linestyle='None', ecolor="g")
#plt.legend(handler_map={lineopt: HandlerLine2D(numpoints=4)})

#lineopt30, = plt.plot(states, optimal_test30_avg, "r", label="Optimal c30%")
#plt.errorbar(states, optimal_test30_avg, optimal_test30_st_dev, linestyle='None', ecolor="r")
#plt.legend(handler_map={lineopt: HandlerLine2D(numpoints=4)})

linesopt, = plt.plot(states, suboptimal_test, "b--", label="Sub-optimal")
#plt.errorbar(states, suboptimal_test, suboptimal_stdev, linestyle='None', ecolor="b")
#plt.legend(handler_map={linesopt: HandlerLine2D(numpoints=4)})

#linesopt, = plt.plot(states, other_test, "c-", label="Sub-optimal")


plt.legend(loc=4)

plt.show()