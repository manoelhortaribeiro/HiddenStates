import matplotlib.pyplot as plt
import matplotlib
from matplotlib.legend_handler import HandlerLine2D
from math import sqrt
import numpy
from numpy import array
SPINE_COLOR = 'gray'

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\usepackage{gensymb}'],
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'text.fontsize': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)

latexify()

svmiter = [5,7,10,25,50]

opt30 = [ 0.82401928, 0.81885452,  0.82772404, 0.81214044, 0.82713726]
sopt30 = [ 0.81749573, 0.8137963,  0.82779987, 0.82177907, 0.83978979]


opt20 = [ 0.82789234, 0.83793004,  0.83277551, 0.83062958, 0.83062958]
sopt20 = [ 0.82000899, 0.8137963,  0.82069985, 0.81778646, 0.81503934]



plt.xlabel("Number of Latent SVM iteractions")
plt.ylabel("Accuracy")

lineopt, = plt.plot(svmiter, opt30, "g-", label="our 30")

lineopt, = plt.plot(svmiter, sopt30, "g--", label="LDCRF 30")

lineopt, = plt.plot(svmiter, opt20, "b-", label="our 20")

lineopt, = plt.plot(svmiter, sopt20, "b--", label="LDCRF 20")


plt.show()