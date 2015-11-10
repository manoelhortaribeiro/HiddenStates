import matplotlib.pyplot as plt

import numpy
from numpy import array
SPINE_COLOR = 'gray'

import latexif
import pprint
latexif.latexify()

states = [6, 10, 14, 18, 22, 26, 30]

arbitrary = []

arbitrary.append(array([ 0.90588912,  0.92699383,  0.93286083,  0.93621147,  0.93991812,
        0.94800092,  0.94041291]))  # seed 1
arbitrary.append(array([ 0.90588912,  0.92289503,  0.92205703,  0.93363083,  0.94620915,
        0.94814196,  0.95016242]))  # seed 1
arbitrary.append(array([ 0.90588912,  0.92387445,  0.93219905,  0.93079083,  0.93963439,
        0.94595777,  0.94041587]))  # seed 1

correlation = []

correlation.append(array([ 0.90588912,  0.91371941,  0.91627386,  0.93249511,  0.93928062,
        0.94152061,  0.94645912]))  # seed 1
correlation.append(array([ 0.90588912,  0.91188985,  0.92351398,  0.93319545,  0.94277467,
        0.93886093,  0.94613498]))  # seed 1
correlation.append(array([ 0.90588912,  0.91728541,  0.9315417 ,  0.92582981,  0.93310589,
        0.94467013,  0.94426692]))  # seed 1

cosine = []

cosine.append(array([ 0.90588912,  0.91701792,  0.91772423,  0.93805965,  0.93567749,
        0.94137566,  0.95048514]))  # seed 1
cosine.append(array([ 0.90588912,  0.91923092,  0.93117317,  0.93807661,  0.93925852,
        0.94578884,  0.94415402]))  # seed 1
cosine.append(array([ 0.90588912,  0.91237192,  0.92839415,  0.93992653,  0.94225563,
        0.94668393,  0.94191523]))  # seed 1


cosine = array(cosine).mean(0)
correlation = array(correlation).mean(0)
arbitrary = array(arbitrary).mean(0)


plt.xlabel("Number of Hidden States")
plt.ylabel("Accuracy")

lineopt, = plt.plot(states, cosine, "g-", label="Ours")

linesopt, = plt.plot(states, correlation, "r-", label="Correlation")

linesopt, = plt.plot(states, arbitrary, "b--", label="Arbitrary")


plt.tight_layout()

ax = plt.gca()

ax.grid(True)
plt.legend(loc=4)

plt.show()