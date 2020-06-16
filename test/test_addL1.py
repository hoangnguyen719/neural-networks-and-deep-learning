import os
import sys
sys.path.append("D:/Work/neural-networks-and-deep-learning/src")

import mnist_loader
import network2_1
import numpy as np

training_data, validation_data, test_data =\
    map(list, mnist_loader.load_data_wrapper())
net = network2_1.Network([784, 10])

# No L1
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
# Accuracy unregularized: 8888 / 10000

# ====================
# Accuracy with L1:

# Lambda: 0.0001, accuracy: 8957 / 10000
# Lambda: 0.01, accuracy: 8907 / 10000
# Lambda: 0.1, accuracy: 8882 / 10000
# Lambda: 0.5, accuracy: 8954 / 10000
# Lambda: 1, accuracy: 8277 / 10000
# Lambda: 2, accuracy: 8758 / 10000
# Lambda: 3, accuracy: 8746 / 10000
# Lambda: 4, accuracy: 8757 / 10000
# Lambda: 5, accuracy: 8470 / 10000
# Lambda: 7, accuracy: 8726 / 10000
# Lambda: 10, accuracy: 8716 / 10000

Generally unregularized net performs better
than L1-regularized net. The only L1-lambda that outperforms
unregularized is lambda=0.0001 or lambda=0.01, which implies that
L1-regularization's effect shouldn be very small in order for
the net to improve.
"""