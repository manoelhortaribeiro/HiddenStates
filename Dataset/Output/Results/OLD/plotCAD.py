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

lineopt, = plt.plot(states[:], opt_test[:], "g-", label="Ours")

linesopt, = plt.plot(states[:], sopt_test[:], "b--", label="Arbitrary")

plt.tight_layout()

ax = plt.gca()

ax.grid(True)
plt.legend(loc=1)

plt.savefig("../Results/imgs/CAD120.pdf")