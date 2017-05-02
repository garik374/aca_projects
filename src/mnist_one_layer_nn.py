from __future__ import print_function
import math
import numpy as np


def sigmoid(s):
    return math.exp(s) / (1. + math.exp(s))


def data_transformation(data):
    # crop the images
    data_features = data[:, 5:28, 5:21]
    data_features = data_features.reshape(len(data), -1)
    # normalize
    data_features = data_features / 255
    return data_features


def nn_train(data, labels, epoch, step_size):
    beta = np.zeros(10)
    w = np.random.uniform(-0.05, 0.05, (len(data[0]), 10))
    for epoch_num in range(epoch):
        for index in range(len(data)):
            hl = data[index].dot(w) + beta
            h_activated = np.array([sigmoid(hn) for hn in hl])
            response = h_activated
            grad_w, grad_beta = back_propagation(data[index], response, labels[index])
            w = w - step_size * grad_w
            beta = beta - step_size * grad_beta
    return w, beta


def back_propagation(x, r, t):
    grad_w = np.outer(x, np.array([(r[j] - t[j]) * (1 - r[j]) * r[j] for j in range(len(r))]))
    grad_w = grad_w
    grad_beta = np.array([(r[j] - t[j]) * (1 - r[j]) * r[j] for j in range(len(r))])
    return grad_w, grad_beta


def test(w, beta, test_data, true_labels):

    test_data_features = data_transformation(test_data)
    count = 0
    for example, label in zip(test_data_features, true_labels):
        hl = example.dot(w) + beta
        h_activated = np.array([sigmoid(hn) for hn in hl])
        response = h_activated
        if np.argmax(response) == label:
            count += 1
    return count
