#!/usr/bin/env python3
'''
test_network.py
~~~~~~~~~~~~~~~

A module to execute loading the mnist dataset and training and evaluating the network
'''

import mnist_loader
import network
import network2
import argparse

parser = argparse.ArgumentParser(description="Script to test the Network class")
parser.add_argument("-l", "--layers", type=list, default=[784,30,10], 
                    help="List of layer-sizes", metavar="")
parser.add_argument("-e", "--epochs", type=int, default=30, 
                    help="Number of training epochs", metavar="")
parser.add_argument("-mbs", "--batchSize", type=int, default=10, 
                    help="Number of training examples per mini-batch", metavar="")
parser.add_argument("-eta", type=float, default=0.5, help="Learning rate", metavar="")
parser.add_argument("-lmbda", type=float, default=4.0, help="Regularization factor", metavar="")
parser.add_argument("-momentum", type=bool, default=True, help="Momentum Factor", metavar="")
parser.add_argument("-mu", type=float, default=0.1, help="Momentum Factor", metavar="")
parser.add_argument("-dropout", type=float, default=0.1, help="Dropout Rate", metavar="")

## Choosing Hyperparameters
# eta - change by multiplying/dividing by 10, find first value, st cost initially decreases, half that
#     - schedule: half after no improvement after 10 epochs
#
# epochs - early stop: stop if no improvement after 10, 20, ..
#
# lmbda - start with 0, find good eta, set lambda=1.0, increase/decrease by factor 10 until good
#
# mini_batch_size - plot validation accuravy over time, pick the one with fastest improvement



args = parser.parse_args()

# load the data, 60k training examples, 10k test examples
training_data, validation_data, test_data = mnist_loader.load_data(50000,10000)

# initialize the network (input layer as size of mnist images, output as confidence)
#net = network.Network(args.layers)
net = network2.Network(args.layers)

# train the network using stochastic gradient descent (backpropagation)
# (training_data, epochs, mini_batch_size, eta, test_data)
#net.SGD(training_data, args.epochs, args.batchSize, args.eta, test_data)
net.SGD(training_data, cost=network2.CrossEntropyCost, regularization=network2.l2_weight_decay,
        epochs=args.epochs, mini_batch_size=args.batchSize, eta=args.eta, lmbda=args.lmbda, 
        momentum=args.momentum, mu=args.mu, dropout_rate=args.dropout, val_data=validation_data)