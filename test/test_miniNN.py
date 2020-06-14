import os
import sys
sys.path.append("D:/Work/neural-networks-and-deep-learning/src")

import mnist_loader
import network2

training_data, validation_data, test_data =\
    map(list, mnist_loader.load_data_wrapper())
net = network2.Network([784, 10])
net.SGD(training_data[:1000], 30, 10, 10, lmbda=5,\
    evaluation_data=validation_data[:100],
    monitor_evaluation_accuracy=True)