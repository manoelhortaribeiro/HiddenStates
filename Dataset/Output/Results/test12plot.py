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

optimal_test.append(array([ 0.8291791 ,  0.82635678,  0.82946598,  0.83315765,  0.83188394,
        0.82595105,  0.82935693,  0.82072343,  0.79597971,  0.79684206,
        0.82508346,  0.82333196,  0.80349009]))  # seed 10

optimal_test.append(array([ 0.82789693,  0.83141112,  0.8333711,  0.83149333,  0.83276599,
                            0.82274169,  0.8369527 ,  0.82068679,  0.8194296,
                            0.81661298,  0.82526975,  0.79316411,  0.80680319 ]))  # seed 5

optimal_test.append(array([ 0.84182568,  0.83384742,  0.82722406,  0.84609953,  0.82789693,
        0.82516423,  0.81327696,  0.79821561,  0.81367814,  0.82218347,
        0.81827299,  0.82723463,  0.82661933]))  # seed 50

optimal_test.append(array([ 0.83550222,  0.83968539,  0.82859233,  0.82606535,  0.83005307,
                            0.82741987,  0.82283307,  0.81029974,  0.80764082,
                            0.81048607,  0.82168738,  0.81766341,  0.82478279]))  # seed 6

optimal_test.append(array([ 0.82372891,  0.83899807,  0.83122976,  0.83608048,  0.83443066,
        0.82430154,  0.82828781,  0.79985172,  0.80434327,  0.83822221,
        0.82429625,  0.83209378,  0.82457291]))  # seed 70




suboptimal_test = []
suboptimal_test.append(array([ 0.83802142,  0.82169266,  0.8181644 ,  0.82304228,  0.81816541,
                               0.81553116,  0.8137963 ,  0.79275656,  0.83658144,
                               0.80710911,  0.82283271,  0.80528918,  0.81804971])) # seed 1

suboptimal_test.append(array([ 0.8291791,  0.83580281,  0.81876983,  0.82030998,  0.82039319,
        0.81975141,  0.81699155,  0.79008207,  0.81078632,  0.82964765,
        0.80641887,  0.81424514,  0.82069103])) # seed 10

suboptimal_test.append(array([ 0.82789693,  0.81380053,  0.82246079,  0.81220894,  0.80981426,
                               0.81069137,  0.81509258,  0.8209821 ,  0.81914066,
                               0.80238884,  0.82187724,  0.81068146,  0.83257833])) # seed 5

suboptimal_test.append(array([ 0.83637548,  0.82411489,  0.81116912,  0.81302296,  0.8030949 ,
        0.82158788,  0.7921938 ,  0.8293516 ,  0.805546  ,  0.81145845,
        0.80988804,  0.80684409,  0.828671  ])) # seed 50

suboptimal_test.append(array([ 0.83121954 ,  0.83434312,  0.83005797,  0.81448261,  0.81875184,
                               0.81303174,  0.80454541,  0.81603647,  0.8276933,
                               0.81390213,  0.82799464,  0.81428004,  0.80570447])) # seed 6

suboptimal_test.append(array([ 0.833357  ,  0.80620476,  0.81749542,  0.82392577,  0.82115957,
        0.81565573,  0.80785991,  0.82956614,  0.82546763,  0.81425008,
        0.82788811,  0.83840675,  0.81006126])) # seed 70


optimal_test = array(optimal_test).mean(0)
suboptimal_test = array(suboptimal_test).mean(0)

print optimal_test


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