import matplotlib.pyplot as plt

import numpy
from numpy import array
SPINE_COLOR = 'gray'

import latexif
import pprint
#latexif.latexify()

states = [48, 42, 36, 30, 24, 18, 12, 6]

arbitrary = []

arbitrary.append(array([ 0.63129163,  0.66858605, 0.67378368, 0.62510438,
                         0.58820275,  0.58319109,  0.50442872,  0.20143244]))  # seed 1

arbitrary.append(array( [0.63421717, 0.63820677, 0.64477087, 0.61066447,
                         0.61300221,  0.55307995,  0.47815937,  0.20619404]))  # seed 2

arbitrary.append(array([  0.63178273, 0.62324567, 0.63423899, 0.61559671,
                          0.62169793,  0.59199289,  0.50217412,  0.2056543]))  # seed 3

correlation = []

correlation.append(array([ 0.59676031,  0.57951379,  0.58017263,  0.53591397,  0.20717255]))  # seed 1
correlation.append(array([ 0.62711334,  0.57247285,  0.57627877,  0.50048226,  0.21036323]))  # seed 2
correlation.append(array([ 0.60904804,  0.63052835,  0.5996046 ,  0.50108224,  0.22196802]))  # seed 3

cosine = []

cosine.append(array([ 0.61334114, 0.66566177, 0.62399199, 0.57394243,
                      0.58155323,  0.57474409,  0.5442654 ,  0.20390654]))  # seed 1

cosine.append(array([ 0.64247373, 0.63467661, 0.61577255, 0.59734093,
                      0.56855122,  0.52730152,  0.54708501,  0.21476108]))  # seed 2

cosine.append(array([ 0.6302896, 0.64150081,  0.65056799, 0.61053246,
                      0.58108265,  0.6164102 ,  0.48720989,  0.21179819]))  # seed 3


euclidian = []

euclidian.append(array([ 0.62969643,  0.57075676,  0.55856037,  0.52958928,  0.20901503]))  # seed 1
euclidian.append(array([ 0.63024278,  0.58693004,  0.58188393,  0.55471561,  0.23177235]))  # seed 2
euclidian.append(array([ 0.64234367,  0.60324327,  0.60345698,  0.51513683,  0.21613778]))  # seed 3


cosine = array(cosine).mean(0)
correlation = array(correlation).mean(0)
arbitrary = array(arbitrary).mean(0)

print arbitrary
euclidian = array(euclidian).mean(0)



plt.xlabel("Number of Hidden States")
plt.ylabel("Accuracy")

lineopt, = plt.plot(states, cosine, "g-", label="Cosine")

#linesopt, = plt.plot(states, correlation, "r-", label="Correlation")

linesopt, = plt.plot(states, arbitrary, "b--", label="Arbitrary")


#linesopt, = plt.plot(states, euclidian, "c-", label="Euclidian")


plt.tight_layout()

ax = plt.gca()

ax.grid(True)
plt.legend(loc=4)

plt.show()