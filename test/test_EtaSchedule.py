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

# No eta scheduling
print("="*20)
net = network2_1.Network([784, 10])
t = dt.now()
net.SGD(training_data[:30000], 50, 10, 3
    , lmbda=5, L1_ratio=0)
print("Accuracy without eta scheduling: {} / {}".format(
    net.accuracy(validation_data), len(validation_data)
    ))
timer(t)

# Eta scheduling
print("\n"+"="*20)
eta_epochs = [5, 7, 10, 15]
print("Accuracy with ETA scheduling: \n")
for e in eta_epochs:
    net = network2_1.Network([784, 10])
    t = dt.now()
    net.SGD(training_data[:30000], 50, 10, 3
        , lmbda=5, L1_ratio=0
        , evaluation_data=validation_data
        , eta_sched_e=e
        , eta_sched_f=2
        , eta_stop_f=5
        , early_stopping_e=None
    )
    timer(t)
    print("Scheduling epoch: {}, accuracy: {} / {}".format(
        e, 
        net.accuracy(validation_data),
        len(validation_data)
    ))
    print("-"*10 + "\n")

""" REMARKS:
Below is an example of the output when trained
with 30,000 training examples
(note that the result is not always the same
since the function is stochastic)

# ====================
# Accuracy without eta scheduling: 8642 / 10000
# Timing: 87 seconds

# ====================
# Accuracy with ETA scheduling:

# Timing: 97 seconds
# Scheduling epoch: 5, accuracy: 9140 / 10000
# ----------

# Timing: 98 seconds
# Scheduling epoch: 7, accuracy: 9132 / 10000
# ----------

# Timing: 97 seconds
# Scheduling epoch: 10, accuracy: 8940 / 10000
# ----------

# Timing: 98 seconds
# Scheduling epoch: 15, accuracy: 8866 / 10000
# ----------

Learning schedule seems helpful when the learning rate 
is set too high - decreasing learning rate helps the algorithm
quickly convert to minima (or its proximity). In this case
, eta=3 is possibly too high, therefore epoch=5, which 
decreases learning rate faster than the rest, out-performs
all other options.
"""