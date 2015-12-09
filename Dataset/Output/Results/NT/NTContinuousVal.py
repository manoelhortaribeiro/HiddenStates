import matplotlib.pyplot as plt
from numpy import array
from Util.latexif import latexify, create_table

latexify()

states = [42, 36, 30, 24, 18, 12, 6]

arbitrary = list()

arbitrary.append(array([0.76389268, 0.78760217, 0.78580565, 0.78211365, 0.71188969,
                        0.74666731, 0.47323461]))  # seed 1
arbitrary.append(array([0.78351134, 0.79088577, 0.80352295, 0.7836343, 0.74898394,
                        0.72999765, 0.47326872]))  # seed 2
arbitrary.append(array([0.80505995, 0.81701895, 0.77076798, 0.78276289, 0.78473682,
                        0.70841345, 0.47326872]))  # seed 3

correlation = list()

correlation.append(array([0]))  # seed 3
correlation.append(array([0]))  # seed 2
correlation.append(array([0]))  # seed 1

cosine = list()

cosine.append(array([0.78497883, 0.78108888, 0.76676657, 0.78391826, 0.74374381,
                     0.74361219, 0.47326872]))  # seed 1
cosine.append(array([0.79784075, 0.78784439, 0.80618355, 0.7932932, 0.74726543,
                     0.68870784, 0.47326872]))  # seed 1
cosine.append(array([0.79726583, 0.77742862, 0.79247167, 0.75720231, 0.74854941,
                     0.73281399, 0.47326872]))  # seed 1

euclidian = list()

euclidian.append(array([0.7917195, 0.80454879, 0.78400533, 0.786413, 0.7804723,
                        0.73301113, 0.47326872]))  # seed 1
euclidian.append(array([0.8071173, 0.78256902, 0.79279897, 0.79192841, 0.74951616,
                        0.73086777, 0.47326872]))  # seed 1
euclidian.append(array([0.79765965, 0.78340887, 0.76873227, 0.76610901, 0.76600075,
                        0.723623, 0.47326872]))  # seed 1

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

# linesopt, = plt.plot(states, correlation, "r-", label="Correlation")

linesopt, = plt.plot(states, arbitrary, "b--", label="Arbitrary")

linesopt, = plt.plot(states, euclidian, "y--", label="Euclidian")

plt.tight_layout()

ax = plt.gca()

ax.grid(True)
plt.legend(loc=4)

plt.savefig("/home/manoel/Projects/hidden_states_entropy/Dataset/Output/Results/imgs/NTVALCONT.pdf")
