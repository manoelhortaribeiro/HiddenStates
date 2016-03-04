import matplotlib.pyplot as plt
from numpy import array
from Util.latexif import latexify, create_table

latexify()

states = [42, 36, 30, 24, 18, 12, 6]

arbitrary = list()

arbitrary.append(array([0.82046544, 0.85042283, 0.80836305, 0.82922776, 0.81140765,
                        0.74036586, 0.55456859]))  # seed 1
arbitrary.append(array([0.80505995, 0.81701895, 0.77076798, 0.78276289, 0.78473682,
                        0.70841345, 0.47326872]))  # seed 2
arbitrary.append(array([0.83447058, 0.86179232, 0.82397741, 0.80826032, 0.79577383,
                        0.73825192, 0.55411578]))  # seed 3

correlation = list()

correlation.append(array([0.78710584, 0.80727254, 0.83906588, 0.8177947, 0.82564179,
                          0.72664919, 0.55464859]))  # seed 3
correlation.append(array([0.81811584, 0.79101972, 0.81287027, 0.80043625, 0.78742552,
                          0.75407143, 0.55464859]))  # seed 2
correlation.append(array([ 0.83196876,  0.84291296,  0.80337008,  0.83173374,  0.81616657,
                        0.73561171,  0.55411578]))  # seed 2

cosine = list()

cosine.append(array([0.79629466, 0.77479555, 0.81002161, 0.77737592, 0.83142953,
                     0.79519643, 0.55411578]))  # seed 1
cosine.append(array([0.82134429, 0.83990954, 0.77547823, 0.81317013, 0.78122912,
                     0.74927192, 0.55464859]))  # seed 1
cosine.append(array([0.80881318, 0.81018729, 0.82991242, 0.81469789, 0.80791193,
                     0.73768837, 0.55411578]))  # seed 1

euclidian = list()

euclidian.append(array([0.8396353, 0.82588206, 0.83412893, 0.85388659, 0.82207573,
                        0.80644031, 0.55411578]))  # seed 1
euclidian.append(array([0.7917195, 0.80454879, 0.78400533, 0.786413, 0.7804723,
                        0.73301113, 0.47326872]))  # seed 1
euclidian.append(array([0.85532069, 0.84883308, 0.8395082, 0.81661757, 0.82819388,
                        0.77956482, 0.55411578]))  # seed 1

cosinestd = array(cosine).max(0)
correlationstd = array(correlation).max(0)
arbitrarystd = array(arbitrary).max(0)
euclidianstd = array(euclidian).max(0)

std = [cosinestd, correlationstd, euclidianstd, arbitrarystd]

cosine = array(cosine).mean(0)
correlation = array(correlation).mean(0)
arbitrary = array(arbitrary).mean(0)
euclidian = array(euclidian).mean(0)

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

plt.savefig("/home/manoel/Projects/hidden_states_entropy/Dataset/Output/Results/imgs/NTVALDISC.pdf")
