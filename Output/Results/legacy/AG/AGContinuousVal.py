import matplotlib.pyplot as plt
from numpy import array
from Util.latexif import latexify, create_table

latexify()

states = [48, 42, 36, 24, 12, 6]

arbitrary = list()

arbitrary.append(array([0.92727223, 0.92858506, 0.90539978, 0.92204894, 0.90724706,
                        0.90036389]))  # seed 1
arbitrary.append(array([0.91362094, 0.93696241, 0.92039456, 0.92900694, 0.92980984,
                        0.90290714]))  # seed 2
arbitrary.append(array([0.89198618, 0.92017802, 0.93231703, 0.91678333, 0.91150465,
                        0.90290714]))  # seed 3

correlation = list()

correlation.append(array([0.93299986, 0.91364178, 0.92569525, 0.91772576, 0.90316308,
                          0.90036389]))  # seed 3
correlation.append(array([0.928443, 0.9292902, 0.922706, 0.92126716, 0.903342,
                          0.90290714]))  # seed 2
correlation.append(array([0.92664411, 0.91400646, 0.93297453, 0.92370021, 0.91284115,
                          0.90290714]))  # seed 1

cosine = list()

cosine.append(array([0.90350944, 0.90836802, 0.91546039, 0.92720214, 0.91533044,
                     0.90290714]))  # seed 1
cosine.append(array([0.90425755, 0.92876301, 0.90817436, 0.9324175, 0.9047837,
                     0.90290714]))  # seed 1
cosine.append(array([0.92077111, 0.90634676, 0.92514346, 0.90842624, 0.90544449,
                     0.90036389]))  # seed 1

euclidian = list()

euclidian.append(array([0.92940371, 0.92707505, 0.90877133, 0.92614818, 0.90366695,
                        0.90036389]))  # seed 1
euclidian.append(array([0.91030584, 0.91455235, 0.92455899, 0.91824794, 0.9127078,
                        0.90290714]))  # seed 1
euclidian.append(array([0.92568743, 0.913981, 0.91341179, 0.93381426, 0.90889707,
                        0.90290714]))  # seed 1

cosinestd = array(cosine).std(0)
correlationstd = array(correlation).std(0)
arbitrarystd = array(arbitrary).std(0)
euclidianstd = array(euclidian).std(0)

std = [cosinestd, correlationstd, euclidianstd, arbitrarystd]

cosine = array(cosine).max(0)
correlation = array(correlation).max(0)
arbitrary = array(arbitrary).max(0)
euclidian = array(euclidian).max(0)

mean = [cosine, correlation, euclidian, arbitrary]

descs = ['cosine', 'correlation', 'euclidian', 'arbitrary']

create_table(mean, std, states, descs)

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

plt.savefig("/home/manoel/Projects/hidden_states_entropy/Dataset/Output/Results/imgs/AGVALCONT.pdf")
