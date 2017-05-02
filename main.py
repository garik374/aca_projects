from __future__ import print_function

import gzip
import struct

import make_data as md

import numpy as np

from pathlib import Path

from mnist_one_layer_nn import data_transformation
from mnist_one_layer_nn import nn_train
from mnist_one_layer_nn import test


def main():

    # training data
    mnist_train_data = Path('data/train-images.gz')
    if not mnist_train_data.is_file():
        mnist_train_data = md.get_data('train-images-idx3-ubyte.gz', 'data/train-images.gz')
    mnist_train_labels = Path('data/train-labels.gz')
    if not mnist_train_labels.is_file():
        mnist_train_labels = md.get_data('train-labels-idx1-ubyte.gz', 'data/train-labels.gz')

    # test data
    mnist_test_data = Path('data/test-images.gz')
    if not mnist_test_data.is_file():
        mnist_test_data = md.get_data('t10k-images-idx3-ubyte.gz', 'data/test-images.gz')

    mnist_test_labels = Path('data/test-labels.gz')
    if not mnist_test_labels.is_file():
        mnist_test_labels = md.get_data('t10k-labels-idx1-ubyte.gz', 'data/test-labels.gz')

    # Read labels file into labels
    with gzip.open(mnist_train_labels, 'rb') as in_gzip:
        magic, num = struct.unpack('>II', in_gzip.read(8))
        labels = struct.unpack('>60000B', in_gzip.read(60000))

    # Read data file into numpy matrices
    with gzip.open(mnist_train_data, 'rb') as in_gzip:
        magic, num, rows, columns = struct.unpack('>IIII', in_gzip.read(16))
        data = np.array([np.reshape(struct.unpack('>{}B'.format(rows * columns),
                                                  in_gzip.read(rows * columns)),
                                    (rows, columns))
                         for _ in range(60000)])

    identity = np.identity(10)
    one_hot_labels = [identity[label] for label in labels]

    data_features = data_transformation(data)

    w, beta = nn_train(data_features, one_hot_labels, 3, 0.05)

    # Read test labels file into labels
    with gzip.open(mnist_test_labels, 'rb') as in_gzip:
        magic, num = struct.unpack('>II', in_gzip.read(8))
        true_labels = struct.unpack('>10000B', in_gzip.read(10000))

    # Read test data file into numpy matrices
    with gzip.open(mnist_test_data, 'rb') as in_gzip:
        magic, num, rows, columns = struct.unpack('>IIII', in_gzip.read(16))
        test_data = np.array([np.reshape(struct.unpack('>{}B'.format(rows * columns),
                                                       in_gzip.read(rows * columns)),
                                         (rows, columns))
                              for _ in range(10000)])

    print("Final result", test(w, beta, test_data, true_labels))

if __name__ == '__main__':
    main()