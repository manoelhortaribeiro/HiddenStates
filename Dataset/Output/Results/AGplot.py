import matplotlib.pyplot as plt

import numpy
from numpy import array
SPINE_COLOR = 'gray'

import latexif

latexif.latexify()

states = [6, 8, 10, 12, 14, 18, 22, 24, 30]

optimal_test = []
optimal_test.append(array([ 0.9049292 ,  0.90990781,  0.91575178,  0.91919491,  0.9202868 ,
        0.93753164,  0.94019407,  0.94201182,  0.94617944]))  # seed 1

optimal_test.append(array([ 0.9049292 ,  0.90253325,  0.91119144,  0.9212405 ,  0.92612842,
        0.93475822,  0.93699933,  0.94477985,  0.94640768]))  # seed 3

optimal_test.append(array([ 0.9049292 ,  0.9040255 ,  0.91409404,  0.92577236,  0.93313452,
        0.93606111,  0.94049705,  0.94205947,  0.94662216]))   # seed 6

optimal_test.append(array([ 0.9049292 ,  0.90433517,  0.9085026 ,  0.92710932,  0.93422968,
        0.937084  ,  0.94412959,  0.94518641,  0.94975143]))  # seed 7


suboptimal_test = []
suboptimal_test.append(array([ 0.9049292 ,  0.90880417,  0.91652772,  0.92334171,  0.92310744,
        0.94447217,  0.93794476,  0.94396284,  0.94606755])) # seed 1

suboptimal_test.append(array([ 0.9049292 ,  0.90433517,  0.9085026 ,  0.92710932,  0.93422968,
        0.937084  ,  0.94412959,  0.94518641,  0.94975143])) # seed 3

suboptimal_test.append(array([ 0.9049292 ,  0.90408748,  0.92537913,  0.92622674,  0.91633071,
        0.93078465,  0.9402849 ,  0.93764743,  0.94699193])) # seed 6

suboptimal_test.append(array([ 0.9049292 ,  0.90408748,  0.92537913,  0.92622674,  0.91633071,
        0.93078465,  0.9402849 ,  0.93764743,  0.94699193])) # seed 7

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

plt.savefig("../Results/imgs/AG.pdf")