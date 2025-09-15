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
parser.add_argument("-lmbda", type=float, default=1.0, help="Regularization factor", metavar="")

args = parser.parse_args()

# load the data, 60k training examples, 10k test examples
training_data, test_data = mnist_loader.load_data(6/7)

# initialize the network (input layer as size of mnist images, output as confidence)
#net = network.Network(args.layers)
net = network2.Network(args.layers, regularization=network2.l1_weight_decay)

# train the network using stochastic gradient descent (backpropagation)
# (training_data, epochs, mini_batch_size, eta, test_data)
#net.SGD(training_data, args.epochs, args.batchSize, args.eta, test_data)
net.SGD(training_data, args.epochs, args.batchSize, args.eta, args.lmbda, test_data)