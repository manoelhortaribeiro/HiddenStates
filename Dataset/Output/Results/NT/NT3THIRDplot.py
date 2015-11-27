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

correlation.append(array([ 0.82549562,  0.83930018,  0.83700096,  0.75741658,  0.59615377]))  # seed 1
correlation.append(array([ 0.8499284 ,  0.82099401,  0.779156  ,  0.772531  ,  0.59615377]))  # seed 2
correlation.append(array([ 0.83903114,  0.82924404,  0.83761767,  0.73075162,  0.59660132]))  # seed 2


cosine = []

cosine.append(array([ 0.83841491,  0.84591811,  0.8118789 ,  0.74920038,  0.59660132]))  # seed 1

cosine.append(array([ 0.82915562,  0.81643854,  0.80826614,  0.76391832,  0.59660132]))  # seed 2

cosine.append(array([ 0.81893405,  0.84412979,  0.84383416,  0.7836016 ,  0.59660132]))  # seed 3

euclidian = []


euclidian.append(array([ 0.84731472,  0.81952398,  0.82177568,  0.75546491,  0.59660132]))  # seed 1
euclidian.append(array([ 0.85382129,  0.81674277,  0.82841133,  0.80067718,  0.59660132]))  # seed 2
euclidian.append(array([ 0.83291696,  0.86264855,  0.8321376 ,  0.7820089 ,  0.59660132]))  # seed 3

cosinestd = array(cosine).std(0)
correlationstd = array(correlation).std(0)
arbitrarystd = array(arbitrary).std(0)
euclidianstd = array(euclidian).std(0)

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


linesopt, = plt.plot(states, euclidian, "c-", label="Euclidian")


plt.tight_layout()

ax = plt.gca()

ax.grid(True)
plt.legend(loc=4)

plt.show()
