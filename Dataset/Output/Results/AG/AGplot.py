import matplotlib.pyplot as plt

import numpy
from numpy import array

SPINE_COLOR = 'gray'

import pprint


states = [6, 10, 14, 18, 22, 26, 30, 35, 40, 45, 50]


arbitrary = []

arbitrary.append(array([0.90588912, 0.92699383, 0.93286083, 0.93621147, 0.93991812,
                        0.94800092, 0.94041291, 0.94791901, 0.9477754, 0.95285628, 0.9489175]))  # seed 1
arbitrary.append(array([0.90588912, 0.92289503, 0.92205703, 0.93363083, 0.94620915,
                        0.94814196, 0.95016242, 0.9504511, 0.94181705, 0.9475161, 0.95322132]))  # seed 2
arbitrary.append(array([0.90588912, 0.92387445, 0.93219905, 0.93079083, 0.93963439,
                        0.94595777, 0.94041587, 0.94808309, 0.94877407, 0.95185718, 0.94974933]))  # seed 3

correlation = []

correlation.append(array([0.90588912, 0.91371941, 0.91627386, 0.93249511, 0.93928062,
                          0.94152061, 0.94645912, 0.95388978, 0.94948898, 0.94692905, 0.94953017]))  # seed 1
correlation.append(array([0.90588912, 0.91188985, 0.92351398, 0.93319545, 0.94277467,
                          0.93886093, 0.94613498, 0.94733247, 0.94508543, 0.94980046, 0.95172796]))  # seed 2
correlation.append(array([0.90588912, 0.91728541, 0.9315417, 0.92582981, 0.93310589,
                          0.94467013, 0.94426692, 0.94569674, 0.94543805, 0.94741819, 0.95011636]))  # seed 3

cosine = []

cosine.append(array([0.90588912, 0.91701792, 0.91772423, 0.93805965, 0.93567749,
                     0.94137566, 0.95048514, 0.9436897, 0.95379143, 0.94907758, 0.95286273]))  # seed 1
cosine.append(array([0.90588912, 0.91923092, 0.93117317, 0.93807661, 0.93925852,
                     0.94578884, 0.94415402, 0.94892961, 0.94443258, 0.94761597, 0.95118325]))  # seed 1
cosine.append(array([0.90588912, 0.91237192, 0.92839415, 0.93992653, 0.94225563,
                     0.94668393, 0.94191523, 0.94900374, 0.95087479, 0.94639483, 0.94470775]))  # seed 1

euclidian = []

euclidian.append(array([0.90588912, 0.92019434, 0.9235711, 0.93636967, 0.93193848,
                        0.93849892, 0.94145999, 0.94996026, 0.94272726, 0.95177286, 0.95301446]))  # seed 1
euclidian.append(array([0.90588912, 0.91597391, 0.93260202, 0.9308337, 0.94563359,
                        0.94447607, 0.93806153, 0.94259824, 0.95106202, 0.95386015, 0.95668299]))  # seed 1
euclidian.append(array([0.90588912, 0.91793681, 0.93142436, 0.93690183, 0.94742124,
                        0.94408173, 0.94494681, 0.95071651, 0.94694151, 0.95186995, 0.95147378]))  # seed 1

cosinestd = array(cosine).std(0)
cosine = array(cosine).max(0)
correlation = array(correlation).max(0)
arbitrary = array(arbitrary).max(0)
euclidian = array(euclidian).max(0)

print euclidian - arbitrary
plt.xlabel("Number of Hidden States")
plt.ylabel("Accuracy")

lineopt, = plt.plot(states, cosine, "g-", label="Cosine")

linesopt, = plt.plot(states, correlation, "r-", label="Correlation")

linesopt, = plt.plot(states, arbitrary, "b--", label="Arbitrary")

linesopt, = plt.plot(states, euclidian, "y--", label="Euclidian")

plt.tight_layout()

ax = plt.gca()

ax.grid(True)
plt.legend(loc=4)

plt.show()
