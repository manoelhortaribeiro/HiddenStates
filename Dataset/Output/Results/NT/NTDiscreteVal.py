import matplotlib.pyplot as plt
from numpy import array
from Util.latexif import latexify, create_table

#latexify()

states = [42, 36, 30, 24, 18, 12, 6]

arbitrary = list()

arbitrary.append(array([ 0.82046544,  0.85042283,  0.80836305,  0.82922776,  0.81140765,
        0.74036586,  0.55456859]))  # seed 1
arbitrary.append(array([ 0.80505995,  0.81701895,  0.77076798,  0.78276289,  0.78473682,
        0.70841345,  0.47326872]))  # seed 2
arbitrary.append(array([ 0.83447058,  0.86179232,  0.82397741,  0.80826032,  0.79577383,
        0.73825192,  0.55411578]))  # seed 3

correlation = list()

correlation.append(array([ 0.93299986,  0.91364178,  0.92569525,  0.91772576,  0.90316308,
        0.90036389]))  # seed 3
correlation.append(array([ 0.928443  ,  0.9292902 ,  0.922706  ,  0.92126716,  0.903342  ,
        0.90290714]))  # seed 2
correlation.append(array([ 0.92664411,  0.91400646,  0.93297453,  0.92370021,  0.91284115,
        0.90290714]))  # seed 1

cosine = list()

cosine.append(array([ 0.90350944,  0.90836802,  0.91546039,  0.92720214,  0.91533044,
        0.90290714]))  # seed 1
cosine.append(array([ 0.90425755,  0.92876301,  0.90817436,  0.9324175 ,  0.9047837 ,
        0.90290714]))  # seed 1
cosine.append(array([ 0.92077111,  0.90634676,  0.92514346,  0.90842624,  0.90544449,
        0.90036389]))  # seed 1

euclidian = list()

euclidian.append(array([ 0.8396353 ,  0.82588206,  0.83412893,  0.85388659,  0.82207573,
        0.80644031,  0.55411578]))  # seed 1
euclidian.append(array([ 0.7917195 ,  0.80454879,  0.78400533,  0.786413  ,  0.7804723 ,
        0.73301113,  0.47326872]))  # seed 1
euclidian.append(array([ 0.85532069,  0.84883308,  0.8395082 ,  0.81661757,  0.82819388,
        0.77956482,  0.55411578]))  # seed 1

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

#lineopt, = plt.plot(states, cosine, "g-", label="Cosine")

#linesopt, = plt.plot(states, correlation, "r-", label="Correlation")

linesopt, = plt.plot(states, arbitrary, "b--", label="Arbitrary")

linesopt, = plt.plot(states, euclidian, "y--", label="Euclidian")

plt.tight_layout()

ax = plt.gca()

ax.grid(True)
plt.legend(loc=4)

plt.show()
