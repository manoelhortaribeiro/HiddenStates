import matplotlib.pyplot as plt
from numpy import array
from Util.latexif import latexify, create_table

latexify()

states = [48, 42, 36, 24, 12, 6]

arbitrary = list()

arbitrary.append(array([0.9448021, 0.94746849, 0.94625136, 0.94141354, 0.92388907,
                        0.89931636]))  # seed 1
arbitrary.append(array([0.94298401, 0.94686284, 0.94329948, 0.94290053, 0.9143308,
                        0.89931636]))  # seed 2
arbitrary.append(array([0.9362573, 0.941009, 0.93965187, 0.93482498, 0.92767013,
                        0.89931636]))  # seed 3

correlation = list()

correlation.append(array([0.94699666, 0.93802259, 0.95007431, 0.9335642, 0.92019815,
                          0.89931636]))  # seed 3
correlation.append(array([0.93977866, 0.94206169, 0.94109411, 0.93115431, 0.92757118,
                          0.89931636]))  # seed 2
correlation.append(array([0.95048273, 0.9362474, 0.94992692, 0.93992619, 0.92833711,
                          0.89931636]))  # seed 1

cosine = list()

cosine.append(array([0.94682783, 0.94633462, 0.94565471, 0.94108713, 0.91855517,
                     0.89931636]))  # seed 1
cosine.append(array([0.9362622, 0.94277828, 0.93212871, 0.94242474, 0.91188547,
                     0.89931636]))  # seed 1
cosine.append(array([0.93975484, 0.94819287, 0.94222811, 0.93950334, 0.92273474,
                     0.89931636]))  # seed 1

euclidian = list()

euclidian.append(array([0.94726828, 0.94111069, 0.94013402, 0.93865708, 0.92114887,
                        0.89931636]))  # seed 1
euclidian.append(array([0.95047621, 0.94621033, 0.94279213, 0.93873412, 0.91947057,
                        0.89931636]))  # seed 1
euclidian.append(array([0.94564619, 0.94505805, 0.93729853, 0.94098695, 0.91613103,
                        0.89931636]))  # seed 1

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

plt.show()
