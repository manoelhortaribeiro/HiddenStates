import matplotlib.pyplot as plt
import matplotlib
from matplotlib.legend_handler import HandlerLine2D
from math import sqrt
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

states = [12, 14, 16, 18, 20, 24, 28, 32, 36]

optimal_test_avg = [0.92990866,  0.93186892,  0.93532716,  0.92496976,  0.92101008,
                    0.92834801,  0.91687232,  0.90997701,  0.90937837]

optimal_test_st_dev = [0.01463371,  0.01853085,  0.00902897,  0.01648409,  0.00816913,
                       0.01791588,  0.01432253,  0.01048111,  0.00661327]

suboptimal_test_avg = [0.92925912,  0.92250197,  0.92602442,  0.92509372,  0.92173483,
                       0.92329653,  0.91027413,  0.91130592,  0.91307202]

suboptimal_test_st_dev = [0.0145285,  0.01059735,  0.01107645,  0.01063729,  0.0060392,
                          0.01779841,  0.01044444,  0.0101253,  0.00817175]

plt.xlabel("Number of Hidden States")
plt.ylabel("Accuracy")

lineopt, = plt.plot(states, optimal_test_avg, "g-", label="Optimal")
plt.errorbar(states, optimal_test_avg, optimal_test_st_dev, linestyle='None', ecolor="g")
plt.legend(handler_map={lineopt: HandlerLine2D(numpoints=4)})

linesopt, = plt.plot(states, suboptimal_test_avg, "b-", label="Sub-optimal")
plt.errorbar(states, suboptimal_test_avg, suboptimal_test_st_dev, linestyle='None', ecolor="b")
plt.legend(handler_map={linesopt: HandlerLine2D(numpoints=4)})

plt.show()