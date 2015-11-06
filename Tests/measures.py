import Util.pyeeg as pyeeg  # Contains the sample entropy calculation
import numpy as np

__author__ = 'Manoel Ribeiro'



def sample_entropy(data_x, data_y):
    """
    This function computes an associated value for each label based on the sample entropy of its features.

    It returns a hash where hash[label_nbr] = associated entropy value.

    --- Parameters:

    - data_x: is the set of features appended with the edges of the conditional random field. The features
     and the edges are organized as follows: [(sample1, edges1), (sample2, edges2)...]

     where sample is a matrix:

      -----  Ft1, Ft2, Ft2 ...
      t = 0   ~    ~    ~  ...
      t = 1   ~    ~    ~  ...
      t = 2   ~    ~    ~  ...
      ... ... ... ... ...  ...

      and edges is a matrix:

      [ [~, ~],
        [~, ~],
        ...
      ]

    - data_y: is the set of of labels of each of the equivalent features in data_x. It is a matrix:

    [
    (sample1) [ 3 3 3 ... ]
    (sample2) [ 4 4 4 ... ]
    ...
    ]

    """

    label_hash = {}

    # This loop creates a hash that, for each label L, has a hash that, for each sample in the dataset
    # has a list. It looks very complicated but it is really just a way to separate the data to then
    # calculate the sample entropy for each sample, and then average it based on the sample entropy.

    for sample in range(len(data_x)):

        for features, y in zip(data_x[sample][0],data_y[sample]):

            if label_hash.has_key(y) is False:
                label_hash[y] = {}

            if label_hash[y].has_key(sample) is False:
                label_hash[y][sample] = []

            label_hash[y][sample].append(features)

    samp_entropy_sum = {}

    # Given the transpose of the matrix we have for each sample, in the form:
    #  -----  t=0, t=1, t=2 ... We then calculate the entropy for each of the
    #  Feat1   ~    ~    ~  ... features and sum them. We then average the sum
    #  Feat2   ~    ~    ~  ... of ALL the different samples to give a entropy
    #  Feat3   ~    ~    ~  ... value for a LABEL!
    #  ... ... ... ... ...  ...

    for label in label_hash.keys():

        partial = 0

        for sample in label_hash[label].keys():
            matrix = np.array(label_hash[label][sample]).transpose()

            for i in matrix:
                std = np.std(i)
                partial += pyeeg.samp_entropy(i, 2, 0.2*std)

        samp_entropy_sum[label] = partial/float(len(label_hash[label].keys()))

    # Last but not least, we calculate the relative  entropy of each of the labels.
    # First we sum the value so far associated with each one of the labels and then
    # we normalize all values dividing them by this sum.

    total_entropy = 0

    for key in label_hash.keys():
        total_entropy += samp_entropy_sum[key]

    for key in label_hash.keys():
        samp_entropy_sum[key] /= (float(total_entropy))

    return samp_entropy_sum.items()


def calculate_maximum_hidden(buckets, y):
    # calculate maximum number of hidden states as number of instances
    maximum = []
    for i in range(buckets):
        maximum.append(0)

    for i in y:
        for j in i:
            maximum[j] += 1

    return maximum


def divide_hidden_states_entropy_c(balls, buckets, measure, c, y):

    maximum = calculate_maximum_hidden(buckets, y)

    original_balls = balls

    # initialize each bucket with one ball
    balls -= buckets
    balls_dist = buckets

    states = []
    for i in range(buckets):
        states.append(1)

    # gets measure into array and normalize it
    values = []
    for i, j in measure:
        values.append(j)

    sum_values = np.array(values).sum()
    values = map(lambda x: x/sum_values, values)

    # get actual values in the distribution
    actual_values = [float(x) / balls_dist for x in states]

    # distribute the rest of the buckets
    while balls != 0:
        diff = [a-b for a, b in zip(values, actual_values)]

        # gets the tuple with (x,y) where X,
        # is position, and Y is the value of diff
        tuplediff = sorted(enumerate(diff), key=(lambda k: k[1]))

        # do some bookkeeping
        balls -= 1
        balls_dist += 1

        for i in range(buckets):
            index = tuplediff[-(i+1)][0]
            if states[index] < c * original_balls or i+1 == buckets:
                if states[index] < maximum[index]:
                    states[index] += 1
                    break

        actual_values = [float(x) / balls_dist for x in states]

    return states


def divide_hidden_states_arbitrary(balls, buckets):

    states = []

    # Initialize each state with 0 balls
    for i in range(buckets):
        states.append(0)

    # Add balls to states in order
    pointer = 0
    for i in range(balls):
        states[pointer % buckets] += 1
        pointer += 1

    return states

