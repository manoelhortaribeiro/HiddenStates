import matplotlib.pyplot as plt
import matplotlib
from matplotlib.legend_handler import HandlerLine2D
from math import sqrt
import numpy
from numpy import array
SPINE_COLOR = 'gray'


states = [10, 12, 14, 16, 18]

optimal_test = []

optimal_test.append(array([0.83802142,  0.83521215,  0.8440584,  0.83870311,  0.83023831]))  # seed 1

optimal_test.append(array([ 0.83492108,  0.83472457,  0.83120225,  0.82674281,  0.83132079]))  # seed 3

optimal_test.append(array([0.82789693,  0.83141112,  0.8333711 ,  0.83149333,  0.83276599]))  # seed 5

optimal_test.append(array([0.83550222,  0.83968539,  0.82859233,  0.82606535,  0.83005307]))  # seed 6

optimal_test.append(array([ 0.83121954,  0.83578447,  0.83734432,  0.83842082,  0.8368666 ]))  # seed 7

optimal_test.append(array([ 0.83491614,  0.83549375,  0.83977572,  0.83054773,  0.8333577 ]))  # seed 7

optimal_stdev = []

suboptimal_test = []
suboptimal_test.append(array([0.83802142,  0.82169266,  0.8181644 ,  0.82304228,  0.81816541])) # seed 1

suboptimal_test.append(array([ 0.83492108,  0.83472457,  0.83120225,  0.82674281,  0.83132079])) # seed 3

suboptimal_test.append(array([ 0.82789693,  0.81380053,  0.82246079,  0.81220894,  0.80981426])) # seed 5

suboptimal_test.append(array([ 0.83550222,  0.83968539,  0.82859233,  0.82606535,  0.83005307])) # seed 6

suboptimal_test.append(array([ 0.83121954 ,  0.83434312,  0.83005797,  0.81448261,  0.81875184])) # seed 7

suboptimal_test.append(array([ 0.83491614,  0.83549375,  0.83977572,  0.83054773,  0.8333577 ])) # seed 8

suboptimal_stdev = []



optimal_test = array(optimal_test).mean(0)
suboptimal_test = array(suboptimal_test).mean(0)
optimal_stdev = numpy.sqrt(array(suboptimal_stdev).__pow__(2).mean(0))
suboptimal_stdev = numpy.sqrt(array(suboptimal_stdev).__pow__(2).mean(0))
print optimal_stdev

plt.xlabel("Number of Hidden States")
plt.ylabel("Accuracy")

lineopt, = plt.plot(states, optimal_test, "g-", label="Optimal")

linesopt, = plt.plot(states, suboptimal_test, "b--", label="Sub-optimal")


#linesopt, = plt.plot(states, other_test, "c-", label="Sub-optimal")


plt.legend(loc=4)

plt.show()