import os
import sys
sys.path.append("D:/Work/neural-networks-and-deep-learning/src")

from datetime import datetime as dt
import mnist_loader
import network2_1
import numpy as np

training_data, validation_data, test_data =\
    map(list, mnist_loader.load_data_wrapper())

# Timing function
def timer(t):
    output = dt.now() - t
    print("Timing: {} seconds".format(output.seconds))

# No early stopping
net = network2_1.Network([784, 10])
print("="*20)
t = dt.now()
net.SGD(training_data[:30000], 50, 10, 3
    , lmbda=5, L1_ratio=0)
timer(t)
print("Accuracy without early stopping: {} / {}".format(
    net.accuracy(validation_data), len(validation_data)
    ))

# Early stopping applied
print("\n"+"="*20)
print("Accuracy with Early Stopping: \n")
es_e = [5,10,20,40]
for e in es_e:
    net = network2_1.Network([784, 10])
    t = dt.now()
    net.SGD(training_data[:30000], 50, 10, 3
        , lmbda=5, L1_ratio=0
        , evaluation_data=validation_data
        , early_stopping_e=e)
    timer(t)
    print("n: {}, accuracy: {} / {}".format(
        e, 
        net.accuracy(validation_data),
        len(validation_data)
    ))
    print("-"*10 + "\n")

""" REMARKS:
Below is an example of the output when training
with 30,000 examples.
(note that the result is not always the same
since the function is stochastic)

# ====================
# Timing: 86 seconds
# Accuracy without early stopping: 8628 / 10000

# ====================
# Accuracy with Early Stopping:

# Early stopped at epoch no.12!
# Timing: 24 seconds
# n: 5, accuracy: 7313 / 10000
# ----------

# Early stopped at epoch no.15!
# Timing: 30 seconds
# n: 10, accuracy: 8694 / 10000
# ----------

# No early stopping used!
# Timing: 98 seconds
# n: 20, accuracy: 8840 / 10000
# ----------

# No early stopping used!
# Timing: 99 seconds
# n: 40, accuracy: 8986 / 10000
# ----------

It can be seen that when ``epochs`` is large
early stopping can help improving running time
while ensuring or even improving accuracy (by reducing
overfitting).
"""