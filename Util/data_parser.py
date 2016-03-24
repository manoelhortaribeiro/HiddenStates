import numpy as np
import scipy.io as sio

__author__ = 'Manoel Ribeiro'


def load_data(mat):
    mat = sio.loadmat(mat)

    labels = mat['labels'][0].tolist()
    seqs = mat['seqs'][0].tolist()

    labels = [label.tolist() for label in labels]
    seqs = [seq.tolist() for seq in seqs]

    # if Y > X, get X. if Y > X get Y

    for i in range(len(labels)):

        if len(seqs[i][0]) > len(labels[i][0]):
            for j in range(len(seqs[i])):
                seqs[i][j] =seqs[i][j][:len(labels[i][0])]

        if len(labels[i][0]) > len(seqs[i][0]):
            labels[i][0] = labels[i][0][:len(seqs[i][0])]

    X = []
    for n, line in enumerate(seqs, 1):
        xs = np.array(line)

        X.append((np.transpose(xs), (np.array([list(range(len(xs[0]) - 1)) + (list(range(1, len(xs[0])))),
                 (list(range(1, len(xs[0])))) + list(range(len(xs[0]) - 1))])).astype(np.int16).T))

    Y = []
    for n, line in enumerate(labels, 1):
        ys = np.array(line)[0]
        Y.append(ys[:].astype(np.int16))

    return X, Y

