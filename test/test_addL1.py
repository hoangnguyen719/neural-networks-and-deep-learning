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
net.SGD(training_data[:30000], 30, 10, 3)
print("="*20)
print("Accuracy unregularized: {} / {}\n".format(
    net.accuracy(validation_data), len(validation_data)
    ))

# L1 applied
print("="*20)
lmbdas = [0.0001, 0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10]
print("Accuracy with L1: \n")
for lmbda in lmbdas:
    net = network2_1.Network([784, 10])
    net.SGD(training_data[:30000], 30, 10, 3
        , lmbda=lmbda, L1_ratio=1
    )
    print("Lambda: {}, accuracy: {} / {}".format(
        lmbda, 
        net.accuracy(validation_data),
        len(validation_data)
    ))

""" REMARKS:
Below is an example of the output when trained
with 30,000 training examples
(note that the result is not always the same
since the function is stochastic)

# ====================
# Accuracy unregularized: 8946 / 10000

# ====================
# Accuracy with L1:

# Lambda: 0.0001, accuracy: 8853 / 10000
# Lambda: 0.01, accuracy: 8868 / 10000
# Lambda: 0.1, accuracy: 8972 / 10000
# Lambda: 0.5, accuracy: 8855 / 10000
# Lambda: 1, accuracy: 8691 / 10000
# Lambda: 2, accuracy: 8855 / 10000
# Lambda: 3, accuracy: 8824 / 10000
# Lambda: 4, accuracy: 8692 / 10000
# Lambda: 5, accuracy: 8622 / 10000
# Lambda: 7, accuracy: 8982 / 10000
# Lambda: 10, accuracy: 8458 / 10000

In this case the unregularized net seems to outperform
L1-regularized nets. The only L1-lambda that outperforms
the unregularized, by only a small margin however, is 
lambda=0.1 or lambda=7, but this may be due to stochastic
process.
"""