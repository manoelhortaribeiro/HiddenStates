import matplotlib.pyplot as plt
from numpy import array
from Util.latexif import latexify, create_table

#latexify()

states = [30, 24, 18, 12, 6]

arbitrary = []

arbitrary.append(array([ 0.82294259,  0.82141513,  0.82038043,  0.78403575,  0.59615377]))  # seed 1

arbitrary.append(array([ 0.8378856 ,  0.83896836,  0.80771602,  0.8026215 ,  0.59660132]))  # seed 2

arbitrary.append(array([ 0.8381184 ,  0.84599226,  0.81534056,  0.7664829 ,  0.59615377]))  # seed 3

correlation = []

correlation.append(array([ 0.6426352,  0.66004447,  0.61641077, 0.59676031,
                           0.57951379, 0.58017263, 0.53591397, 0.20717255]))  # seed 1
correlation.append(array([0.62811314,  0.62318541,  0.61577255, 0.62711334,
                          0.57247285, 0.57627877, 0.50048226, 0.21036323]))  # seed 2
correlation.append(array([ 0.65141778,  0.65471109,  0.64607639, 0.60904804,
                           0.63052835, 0.5996046, 0.50108224, 0.22196802]))  # seed 3

cosine = []

cosine.append(array([0.61334114, 0.66566177, 0.62399199, 0.57394243,
                     0.58155323, 0.57474409, 0.5442654, 0.20390654]))  # seed 1

cosine.append(array([0.64247373, 0.63467661, 0.61577255, 0.59734093,
                     0.56855122, 0.52730152, 0.54708501, 0.21476108]))  # seed 2

cosine.append(array([0.6302896, 0.64150081, 0.65056799, 0.61053246,
                     0.58108265, 0.6164102, 0.48720989, 0.21179819]))  # seed 3

euclidian = []


euclidian.append(array([ 0.84731472,  0.81952398,  0.82177568,  0.75546491,  0.59660132]))  # seed 1
euclidian.append(array([ 0.85382129,  0.81674277,  0.82841133,  0.80067718,  0.59660132]))  # seed 2
euclidian.append(array([ 0.83291696,  0.86264855,  0.8321376 ,  0.7820089 ,  0.59660132]))  # seed 3

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

#lineopt, = plt.plot(states, cosine, "g-", label="Cosine")

#linesopt, = plt.plot(states, correlation, "r-", label="Correlation")

linesopt, = plt.plot(states, arbitrary, "b--", label="Arbitrary")


linesopt, = plt.plot(states, euclidian, "c-", label="Euclidian")


plt.tight_layout()

ax = plt.gca()

ax.grid(True)
plt.legend(loc=4)

plt.show()
