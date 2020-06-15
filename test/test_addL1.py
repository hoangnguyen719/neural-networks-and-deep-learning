import os
import sys
sys.path.append("D:/Work/neural-networks-and-deep-learning/src")

import mnist_loader
import network2_1
import numpy as np

training_data, validation_data, test_data =\
    map(list, mnist_loader.load_data_wrapper())

# No L1
net = network2_1.Network([784, 10])
net.SGD(training_data[:1000], 30, 10, 3)
print("="*20)
print("Accuracy unregularized: {} / {}".format(
    net.accuracy(validation_data), len(validation_data)
    ))

# L1 applied
print("="*20)
lmbdas = [0.0001, 0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10]
print("Accuracy with L1: \n")
for lmbda in lmbdas:
    net.SGD(training_data[:1000], 30, 10, 3
        , lmbda=lmbda, L1_ratio=1
    )
    print("Lambda: {}, accuracy: {} / {}".format(
        lmbda, 
        net.accuracy(validation_data),
        len(validation_data)
    ))