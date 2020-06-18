import os
import sys
sys.path.append("D:/Work/neural-networks-and-deep-learning/src")

import mnist_loader
import network2_1
import numpy as np

training_data, validation_data, test_data =\
    map(list, mnist_loader.load_data_wrapper())
net = network2_1.Network([784, 10])

# No eta scheduling
print("="*20)
net.SGD(training_data[:30000], 50, 10, 3
    , lmbda=5, L1_ratio=0)
print("Accuracy without eta scheduling: {} / {}\n".format(
    net.accuracy(validation_data), len(validation_data)
    ))

# Eta scheduling
print("="*20)
eta_epochs = [5, 7, 10, 15]
print("Accuracy with ETA scheduling: \n")
for e in eta_epochs:
    net.SGD(training_data[:30000], 50, 10, 3
        , lmbda=5, L1_ratio=0
        , evaluation_data=validation_data
        , eta_sched_e=e
        , eta_sched_f=2
        , eta_stop_f=5
        , early_stopping_e=None
    )
    print("Scheduling epoch: {}, accuracy: {} / {}\n".format(
        e, 
        net.accuracy(validation_data),
        len(validation_data)
    ))

""" REMARKS:
Below is an example of the output when trained
with 30,000 training examples
(note that the result is not always the same
since the function is stochastic)

# ====================
# Accuracy without eta scheduling: 8578 / 10000

# ====================
# Accuracy with ETA scheduling:

# Early stopped at epoch no.41, eta=0.09375
# Scheduling epoch: 5, accuracy: 9130 / 10000

# Scheduling epoch: 7, accuracy: 9002 / 10000

# Scheduling epoch: 10, accuracy: 8993 / 10000

# Scheduling epoch: 15, accuracy: 8838 / 10000

Learning schedule seems helpful when the learning rate 
is set too high. In this case, eta=3 is possibly too high,
therefore epoch=5, which decreases learning rate faster
than the rest, out-performs all other options.
"""