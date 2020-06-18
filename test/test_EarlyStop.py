import os
import sys
sys.path.append("D:/Work/neural-networks-and-deep-learning/src")

from datetime import datetime as dt
import mnist_loader
import network2_1
import numpy as np

training_data, validation_data, test_data =\
    map(list, mnist_loader.load_data_wrapper())
net = network2_1.Network([784, 10])

# Timing function
def timer(t):
    output = dt.now() - t
    print("Timing: {} seconds".format(output.seconds))

# No early stopping
t = dt.now()
net.SGD(training_data[:30000], 40, 10, 3
    , lmbda=5, L1_ratio=0)
print("="*20)
print("Accuracy without early stopping: {} / {}".format(
    net.accuracy(validation_data), len(validation_data)
    ))
timer(t)


# Early stopping applied
print("\n"+"="*20)
print("Accuracy with Early Stopping: \n")
es_e = [5,10,20,40]
for e in es_e:
    t = dt.now()
    net.SGD(training_data[:30000], 50, 10, 3
        , lmbda=5, L1_ratio=0
        , evaluation_data=validation_data
        , early_stopping_e=e)
    print("n: {}, accuracy: {} / {}".format(
        e, 
        net.accuracy(validation_data),
        len(validation_data)
    ))
    timer(t)
    print("-"*10 + "\n")

""" REMARKS:
Below is an example of the output when training
with 30,000 examples.
(note that the result is not always the same
since the function is stochastic)

# ====================
# Accuracy without early stopping: 8766 / 10000
# Timing: 72 seconds

# ====================
# Accuracy with Early Stopping:

# Early stopped at epoch no.8!
# n: 5, accuracy: 8292 / 10000
# Timing: 17 seconds
# ----------

# Early stopped at epoch no.17!
# n: 10, accuracy: 8780 / 10000
# Timing: 34 seconds
# ----------

# Early stopped at epoch no.32!
# n: 20, accuracy: 8284 / 10000
# Timing: 64 seconds
# ----------

# No early stopping used!
# n: 40, accuracy: 8620 / 10000
# Timing: 98 seconds
# ----------

It can be seen that when ``epochs`` is large
early stopping can help improving running time
while ensuring accuracy.
"""