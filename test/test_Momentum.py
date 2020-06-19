import os
import sys
sys.path.append("D:/Work/neural-networks-and-deep-learning/src")

from datetime import datetime as dt
import mnist_loader
import network2_1
import numpy as np

training_data, validation_data, test_data =\
    map(list, mnist_loader.load_data_wrapper())
def timer(t):
    output = dt.now() - t
    print("Timing: {} seconds".format(output.seconds))

# No momentum
print("="*20)
net = network2_1.Network([784, 10])
t = dt.now()
net.SGD(training_data[:30000], 30, 10, 3
    , lmbda=5, L1_ratio=0)
timer(t)
print("Accuracy momentum: {} / {}".format(
    net.accuracy(validation_data), len(validation_data)
    ))

# With momentum
print("\n"+"="*20)
mus = np.arange(0.1, 1, 0.1)
print("Accuracy with momentum:\n")
for mu in mus:
    net = network2_1.Network([784, 10])
    t = dt.now()
    net.SGD(training_data[:30000], 30, 10, 3
        , lmbda=5, L1_ratio=0
        , mu=mu
    )
    timer(t)
    print("mu: {}, accuracy: {} / {}\n".format(
        mu, 
        net.accuracy(validation_data),
        len(validation_data)
    ))

""" REMARKS:
Below is an example of the output when trained
with 30,000 training examples
(note that the result is not always the same
since the function is stochastic)

# ====================
# Timing: 53 seconds
# Accuracy momentum: 8643 / 10000

# ====================
# Accuracy with momentum:

# Timing: 53 seconds
# mu: 0.1, accuracy: 8476 / 10000

# Timing: 54 seconds
# mu: 0.2, accuracy: 8771 / 10000

# Timing: 54 seconds
# mu: 0.30000000000000004, accuracy: 8831 / 10000

# Timing: 54 seconds
# mu: 0.4, accuracy: 8187 / 10000

# Timing: 54 seconds
# mu: 0.5, accuracy: 8668 / 10000

# Timing: 55 seconds
# mu: 0.6, accuracy: 8564 / 10000

# Timing: 54 seconds
# mu: 0.7000000000000001, accuracy: 8479 / 10000

# D:/Work/neural-networks-and-deep-learning/src\network2_1.py:454: RuntimeWarning: overflow encountered in exp
#   return 1.0/(1.0+np.exp(-z))
# Timing: 54 seconds
# mu: 0.8, accuracy: 8618 / 10000

# Timing: 55 seconds
# mu: 0.9, accuracy: 8014 / 10000

The first visible thing is that higher value of mu creates
greater "velocity", thereby driving the weights and biases
"faster" - to the point that they are two big and created
overflow (since we use exponential function, exp(1000)
is already too big).

In addition, too small or too big mu can create either
slow learning (leading to sub-optimal result) or overfitting,
respectively.

However, having a small enough mu can improve accuracy,
possibly by driving the weights and biases closer
to minimas.
"""