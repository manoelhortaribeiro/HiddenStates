import numpy as np

__author__ = 'Bruno Teixeira'


def remove_activity_data(X, Y, activity):
    Y_f = []
    X_f = []
    # remove activity from data train
    for i, y in enumerate(Y):
        for index in [item for item in range(len(y)) if y[item] == activity]:
            y[index] = False
            X[i][0][index] = False
        Y_f.append(y)
        X_f.append((X[i][0], np.array([range(len(X[i][0]) - 1) + (range(1, len(X[i][0]))), (range(1, len(X[i][0]))) +
                                       range(len(X[i][0]) - 1)]).astype(np.int16).T))
    return X_f, Y_f


def load_data(file_data_train, file_label_train, file_data_test, file_label_test):
    # Open and load Data Train
    f = open(file_data_train)
    data = f.readlines()
    f.close()
    X = []
    xs = []
    for n, line in enumerate(data, 1):
        sline = np.array(line.rstrip().split(','))
        if len(sline) == 1:
            if len(xs) > 1:
                X.append(
                    (np.transpose(xs), (np.array([range(len(xs[0]) - 1) + (range(1, len(xs[0]))), (range(1, len(xs[0])))+range(len(xs[0]) - 1)])).astype(np.int16).T))
                xs = []
        elif len(sline) > 2:
            xs.append(np.array(line.rstrip().split(','))[:].astype(np.float))

    f = open(file_label_train)
    data = f.readlines()
    f.close()

    Y = []
    for n, line in enumerate(data, 1):
        sline = np.array(line.rstrip().split(','))
        if len(sline) > 2:
            Y.append(np.array(line.rstrip().split(','))[:].astype(np.int16))

    # Open and load Data test
    f = open(file_data_test)
    data = f.readlines()
    f.close()
    X_t = []
    xs = []
    for n, line in enumerate(data, 1):
        sline = np.array(line.rstrip().split(','))
        if len(sline) == 1:
            if len(xs) > 1:
                X_t.append(
                    (np.transpose(xs), (np.array([range(len(xs[0]) - 1) + (range(1, len(xs[0]))), (range(1, len(xs[0])))+range(len(xs[0]) - 1)])).astype(np.int16).T))
            xs = []
        elif len(sline) > 2:
            xs.append(np.array(line.rstrip().split(','))[:].astype(np.float))

    f = open(file_label_test)
    data = f.readlines()
    f.close()
    Y_t = []
    for n, line in enumerate(data, 1):
        sline = np.array(line.rstrip().split(','))
        if len(sline) > 2:
            Y_t.append(np.array(line.rstrip().split(','))[:].astype(np.int16))

    return X, Y, X_t, Y_t

