import matplotlib.pyplot as plt

import numpy
from numpy import array
SPINE_COLOR = 'gray'

import latexif

#latexif.latexify()

states = [ 10, 12, 14, 16, 18]

optimal_test = []
optimal_test.append(array([ 0.82828256,  0.82732814,  0.83257476,  0.82311532,  0.81477329]))  # seed 1

optimal_test.append(array([ 0.8241957 ,  0.81826775,  0.81730489,  0.80814959,  0.83852983]))  # seed 2

optimal_test.append(array([0.81913642,  0.83238384,  0.83566947,  0.83130171,  0.8046441]))   # seed 3



suboptimal_test = []
suboptimal_test.append(array( [0.81933546,  0.83258816,  0.81592248,  0.8226563 ,  0.79608057])) # seed 1

suboptimal_test.append(array([ 0.81904011,  0.82557419,  0.821083  ,  0.81263196,  0.81826919])) # seed 2

suboptimal_test.append(array([ 0.80482028,  0.82615988,  0.83362833,  0.81234617,  0.81203675])) # seed 3

stdev = array(optimal_test).std(0)


optimal_test = array(optimal_test).mean(0)

suboptimal_test = array(suboptimal_test).mean(0)


plt.xlabel("Number of Hidden States")
plt.ylabel("Accuracy")

lineopt, = plt.plot(states, optimal_test, "g-", label="Ours")


linesopt, = plt.plot(states, suboptimal_test, "b--", label="Arbitrary")



plt.tight_layout()

ax = plt.gca()

ax.grid(True)
plt.legend(loc=4)

plt.savefig("../Results/imgs/NO9CAD.pdf")