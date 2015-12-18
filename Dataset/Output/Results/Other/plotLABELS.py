import Util.pyeeg as pyeeg  # Contains the sample entropy calculation
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import latexif

def get_labels_features():
    data = sio.loadmat("/home/manoel/Projects/hidden_states_entropy/Dataset/Data/ArmGesture/Discrete3/ArmGestureDiscrete3.mat")

    label_hash = {}
    normalized_hash = {}

    labels = data['labels'].transpose()
    sequences = data['seqs'].transpose()

    # each label is a vector containing the same label for the time length, such as:
    # [0,0,0,0,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1,1,1], [3,3,3,3,3,3,3,3,3,3], but with a bunch
    # of tuples, therefore we use label[0][0][0][0], to get rid of unnecessary tuples
    for label in labels:
        if label_hash.has_key(label[0][0][0]) is False:
            label_hash[label[0][0][0]] = []
            normalized_hash[label[0][0][0]] = []

    # this e
    for idx, label in enumerate(labels):
        label_hash[label[0][0][0]].append(sequences[idx][0])

    old_label_hash = label_hash.copy()

    biggest_len = 0

    for idx, label_samples in label_hash.items():
        for sample in label_samples:

            if len(sample[0]) > biggest_len:
                biggest_len = len(sample[0])

    for idx, label_samples in label_hash.items():

        normalized_samples = []
        for idy, sample in enumerate(label_samples):

            matrix = []

            for idz, feature in enumerate(sample):
                length = len(feature)
                range_norm = np.array(range(1,biggest_len)) * float(length)/biggest_len
                normal_array = np.interp(range_norm, range(length), feature)
                matrix.append(normal_array)

            normalized_samples.append(matrix)

        label_hash[idx] = normalized_samples

    return label_hash, old_label_hash

label_hash, old_label_hash = get_labels_features()


latexif.latexify()

f, axarray = plt.subplots(10, 2, figsize=(12,12))


for feat in range(10):
    for i in old_label_hash[1]:
        axarray[feat, 0].plot(i[feat])


    for i in label_hash[1]:
        axarray[feat,1].plot(i[feat])


plt.tight_layout()

plt.savefig("../Results/imgs/LABEL1.pdf")