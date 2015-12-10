import matplotlib.pyplot as plt
from numpy import array
from Util.latexif import latexify, create_table

latexify()

states = [48, 42, 36, 24, 12, 6]

arbitrary = list()

arbitrary.append(array([0.94633579, 0.96009821, 0.94220507, 0.95731801, 0.94165164,
                        0.94134169]))  # seed 1
arbitrary.append(array([0.93650386, 0.95181988, 0.93767417, 0.95619298, 0.93698095,
                        0.94134169]))  # seed 2
arbitrary.append(array([0.95902129, 0.93815766, 0.94989455, 0.94309823, 0.95116909,
                        0.94134169]))  # seed 3

correlation = list()

correlation.append(array([0.94902947, 0.94276338, 0.95347715, 0.95223367, 0.96056187,
                          0.94134169]))  # seed 3
correlation.append(array([0.94382094, 0.94696648, 0.94989201, 0.94417133, 0.96134891,
                          0.94134169]))  # seed 2
correlation.append(array([0.95086233, 0.94480316, 0.95110747, 0.94992178, 0.95517857,
                          0.94134169]))  # seed 1

cosine = list()

cosine.append(array([0.95195369, 0.9674897, 0.95781736, 0.9492331, 0.95051178,
                     0.94134169]))  # seed 1
cosine.append(array([0.94856042, 0.94665432, 0.94930949, 0.95656793, 0.94927374,
                     0.94134169]))  # seed 1
cosine.append(array([0.95770851, 0.95243731, 0.95782041, 0.94514223, 0.94720714,
                     0.94134169]))  # seed 1

euclidian = list()

euclidian.append(array([0.93239307, 0.95818698, 0.94351515, 0.94597292, 0.94060738,
                        0.94134169]))  # seed 1
euclidian.append(array([0.94787758, 0.93879063, 0.94458887, 0.96127838, 0.93600208,
                        0.94134169]))  # seed 1
euclidian.append(array([0.96200431, 0.95064167, 0.95860683, 0.94546408, 0.94139353,
                        0.94134169]))  # seed 1

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

plt.savefig("/home/manoel/Projects/hidden_states_entropy/Dataset/Output/Results/imgs/AGVALDISC.pdf")
