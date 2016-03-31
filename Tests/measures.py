import numpy as np
import scipy.io as sio
import scipy.spatial.distance as distance
import math

__author__ = 'Manoel Ribeiro'


def normalize_samples(y, x):

    label_hash, normalized_hash = {}, {}
    labels, sequences = y, x

    for label in labels:
        if label_hash.has_key(label[0]) is False:
            label_hash[label[0]] = []
            normalized_hash[label[0]] = []

    # this appends all the samples to each corresponding label
    for idx, label in enumerate(labels):
        label_hash[label[0]].append(sequences[idx][0])

    biggest_len = 0

    for idx, label_samples in label_hash.items():
        for sample in label_samples:
            if len(sample) > biggest_len:
                biggest_len = len(sample)

    # normalize everything to the biggest label
    for idx, label_samples in label_hash.items():
        normalized_samples = []

        for idy, sample in enumerate(label_samples):
            matrix = []
            sample_t = np.transpose(np.array(sample))

            for idz, feature in enumerate(sample_t):
                length = len(feature)
                range_norm = np.array(range(1,biggest_len)) * float(length)/biggest_len
                normal_array = np.interp(range_norm, range(length), feature)
                matrix.append(normal_array)

            normalized_samples.append(matrix)

        normalized_hash[idx] = normalized_samples

    return normalized_hash


def calculate_dist(y, x, dist):

    label_hash = normalize_samples(y, x)


    label_dist = {}
    total_dist = 0
    num_feat = len(x[0][0][0])

    for label in label_hash.keys():
        label_dist[label] = 0
        for feature in range(num_feat):
            feature_dist = 0
            for sample1 in label_hash[label]:
                for sample2 in label_hash[label]:
                    tmp = dist(sample1[feature], sample2[feature])

                    if math.isnan(tmp) is False:
                        feature_dist += tmp
                        total_dist += tmp

            label_dist[label] += feature_dist

    for key in label_dist.keys():

        label_dist[key] /= total_dist
    return label_dist.items()


def calculate_maximum_hidden(buckets, y):
    # calculate maximum number of hidden states as number of instances
    maximum = []
    for i in range(buckets):
        maximum.append(0)

    for i in y:
        for j in i:
            maximum[j] += 1

    return maximum


def divide_hidden_states_measure_c(balls, buckets, measure, c, y):

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


