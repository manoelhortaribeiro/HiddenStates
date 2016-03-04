import matplotlib.pyplot as plt
import matplotlib
from matplotlib.legend_handler import HandlerLine2D
from math import sqrt
import numpy
from numpy import array
SPINE_COLOR = 'gray'
from python_data import *
import latexif


latexif.latexify()


plt.xlabel("Number of Hidden States")
plt.ylabel("Accuracy")

lineopt, = plt.plot(states[0:6], opt_test[0:6], "g-", label="Ours")

linesopt, = plt.plot(states[0:6], sopt_test[0:6], "b--", label="Arbitrary")

plt.tight_layout()

ax = plt.gca()

ax.grid(True)
plt.legend(loc=3)

plt.savefig("../Results/imgs/CAD120part.pdf")
