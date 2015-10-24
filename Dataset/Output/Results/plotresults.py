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

states = [20, 24, 28, 34, 40, 50, 60, 70, 80, 90, 100]

optimal_test_avg = [0.83246115,  0.8218603 ,  0.83647529,  0.81195062,  0.82077249, 0.83707442,
                    0.80028174,  0.81904998,  0.83384388,  0.82120858,  0.83635256]

optimal_test_st_dev = [0.02967843,  0.02458085,  0.02046321,  0.01884786,  0.04443667, 0.02193327,
                       0.0179616 ,  0.02269691,  0.01716422,  0.03211822,  0.03042553]

optimal_test30_avg = [0.83023337,  0.81974224,  0.81349959,  0.81975141,  0.82695132, 0.83794698,
                      0.8144734,  0.8270169,  0.81737793,  0.8198915,  0.83693894]

optimal_test30_st_dev = [0.02392133,  0.00943456,  0.02138163,  0.00351471,  0.01242099, 0.0247848,
                         0.01849568,  0.0354262 ,  0.02789569,  0.03992322,  0.02973785]

suboptimal_test_avg =  [0.81735903,  0.8192368 ,  0.80867533,  0.82586498,  0.83312217, 0.81411102,
                        0.82194776,  0.80611723,  0.80718494,  0.82604736,  0.82957465]

suboptimal_test_st_dev = [0.01728637,  0.02211535,  0.02272138,  0.01631501,  0.01543042, 0.01543365,
                          0.04077927,  0.02427879,  0.01882473,  0.02308896,  0.01611951]


plt.xlabel("Number of Hidden States")
plt.ylabel("Accuracy")

lineopt, = plt.plot(states, optimal_test_avg, "g-", label="Optimal")
#plt.errorbar(states, optimal_test_avg, optimal_test_st_dev, linestyle='None', ecolor="g")
plt.legend(handler_map={lineopt: HandlerLine2D(numpoints=4)})

#lineopt30, = plt.plot(states, optimal_test30_avg, "r", label="Optimal c30%")
#plt.errorbar(states, optimal_test30_avg, optimal_test30_st_dev, linestyle='None', ecolor="r")
#plt.legend(handler_map={lineopt: HandlerLine2D(numpoints=4)})

linesopt, = plt.plot(states, suboptimal_test_avg, "b--", label="Sub-optimal")
#plt.errorbar(states, suboptimal_test_avg, suboptimal_test_st_dev, linestyle='None', ecolor="b")
plt.legend(handler_map={linesopt: HandlerLine2D(numpoints=4)})

plt.legend(loc=4)

plt.show()