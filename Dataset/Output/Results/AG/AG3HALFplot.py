import matplotlib.pyplot as plt
from numpy import array
from Util.latexif import latexify, create_table

#latexify()

states = [48, 42, 36, 24, 12, 6]

arbitrary = list()

arbitrary.append(array([ 0.91664896,  0.9244158 ,  0.92115816,  0.92402469,  0.91611034,
        0.8851862 ]))  # seed 1
arbitrary.append(array([0.92424046,  0.92811207,  0.92135719,  0.92247462,  0.90725761,
        0.8851862 ]))  # seed 2
arbitrary.append(array([ 0.92219096,  0.9144725 ,  0.92004961,  0.91611455,  0.89533022,
        0.8851862 ]))  # seed 3
arbitrary.append(array([ 0.92168564,  0.92587753,  0.9268546 ,  0.91707636,  0.91729544,
        0.8851862 ]))  # seed 4
arbitrary.append(array([ 0.92744852,  0.92470426,  0.92487528,  0.92775154,  0.90893781,
        0.8851862 ]))  # seed 5
arbitrary.append(array([ 0.92078401,  0.92085941,  0.92712922,  0.92286039,  0.90437449,
        0.8851862 ]))  # seed 6

correlation = list()

correlation.append(array([ 0.9237986 ,  0.91862979,  0.9240619 ,  0.91891403,  0.91254182,
        0.8851862 ]))  # seed 3
correlation.append(array([ 0.92731981,  0.91722062,  0.92568536,  0.92979601,  0.90413612,
        0.8851862 ]))  # seed 2
correlation.append(array([ 0.91959725,  0.92269788,  0.92070616,  0.91963103,  0.90974868,
        0.8851862 ]))  # seed 1
correlation.append(array([ 0.91516572,  0.91873267,  0.91895265,  0.9196988 ,  0.8920006 ,
        0.8851862 ]))  # seed 4
correlation.append(array([ 0.91913221,  0.91764584,  0.92211589,  0.92172464,  0.90845149,
        0.8851862 ]))  # seed 5
correlation.append(array([ 0.92091913,  0.91699026,  0.92319348,  0.91158024,  0.90992426,
        0.8851862 ]))  # seed 6

cosine = list()

cosine.append(array([ 0.91393223,  0.92702525,  0.92572313,  0.91867931,  0.89581009,
        0.8851862 ]))  # seed 1
cosine.append(array([ 0.91763546,  0.92527231,  0.9264414 ,  0.92467355,  0.90301959,
        0.8851862 ]))  # seed 1
cosine.append(array([ 0.92046921,  0.92792427,  0.93278641,  0.91667247,  0.90868953,
        0.8851862 ]))  # seed 1
cosine.append(array([ 0.91282838,  0.91828797,  0.92755658,  0.91856719,  0.89665437,
        0.8851862 ]))  # seed 4
cosine.append(array([ 0.92009801,  0.91597438,  0.92503669,  0.91263271,  0.90753053,
        0.8851862 ]))  # seed 5
cosine.append(array([ 0.92288389,  0.91909325,  0.92298241,  0.93319326,  0.90808911,
        0.8851862 ]))  # seed 6

euclidian = list()

euclidian.append(array([ 0.92268064,  0.92118606,  0.92051044,  0.91398197,  0.89737139,
        0.8851862 ]))  # seed 1
euclidian.append(array([ 0.91998935,  0.92946603,  0.92495728,  0.90828459,  0.90182959,
        0.8851862 ]))  # seed 1
euclidian.append(array([ 0.91365574,  0.91962018,  0.92838144,  0.93052068,  0.90121795,
        0.8851862 ]))  # seed 1
euclidian.append(array([ 0.92837798,  0.92497025,  0.93249763,  0.91103146,  0.89431666,
        0.8851862 ]))  # seed 4
euclidian.append(array([ 0.92281323,  0.92564151,  0.92419557,  0.92174853,  0.91858065,
        0.8851862 ]))  # seed 5
euclidian.append(array([ 0.91900502,  0.91891175,  0.92310453,  0.92376069,  0.90896643,
        0.8851862 ]))  # seed 6

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
