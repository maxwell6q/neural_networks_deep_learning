'''
network2.py
~~~~~~~~~~~

A module to implement the stochastic gradient descent learning algorithm for a
feedforward neural network. Gradients are calculated using backpropagation. 
This script also includes several improvements to the vanilla version, such as 
a different cost functions, regularization and initialization.
'''

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        '''
        Returns the cross entropy cost for one input dataum. 
        "a" is the network output, "y" is the desired output.
        '''
        return np.sum(np.nan_to_num(np.multiply(y, np.log(a)) + 
                                    np.multiply((1-y), np.log(1-a))))
    
    @staticmethod
    def delta(z, a, y):
        '''
        Returns the error delta for the output layer. 
        "z" is not used, and only included to provide consistent interfacing.
        '''
        return (a-y)
    

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        '''
        Returns the quadratic cost for one input datum.
        "a" is the network output, "y" is the desired output.
        '''
        return 0.5*np.linalg.norm(a-y)**2
    
    @staticmethod
    def delta(z, a, y):
        '''Returns the error delta for the output layer.'''
        return np.multiply((a-y), sigmoid_prime(z))



def l2_weight_decay(w, lmbda, eta, n):
    return w*(1 - (lmbda*eta)/n)


def l1_weight_decay(w, lmbda, eta, n):
    return w - lmbda*eta/n * np.sign(w)


def no_weight_decay(w, lmbda, eta, n):
    return w



class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost, regularization=l2_weight_decay):
        '''
        num_layers - number of layers, including input and output
        sizes  - list containing the number of neurons per layer
        biases - list containing the bias vectors for each layer
        weigts - list containing the weight matrices for each layer
                 size(weights[i]) = (sizes[i], sizes[i-1])
                 both biases and weights are lists of np.arrays
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.defaultWeightInitializer()
        self._weight_decay = regularization
        
        self.score = 0
        self.best_biases = [np.zeros((nex, 1)) for nex in sizes[1:]]
        self.best_weights = [np.zeros((nex, prev))
                             for nex, prev in zip(sizes[1:], sizes[:-1])]


    def defaultWeightInitializer(self):
        '''
        Initializes the biases with unit variance and the 
        weights with small variance to prevent saturation.
        '''
        self.biases = [np.random.randn(nex, 1) for nex in self.sizes[1:]]
        self.weights = [np.random.randn(nex, prev)/np.sqrt(prev)
                        for nex, prev in zip(self.sizes[1:], self.sizes[:-1])]
        

    def largeWeigtInitializer(self):
        '''Initializes the biases and weights with unit variance.'''
        self.biases = [np.random.randn(nex, 1) for nex in self.sizes[1:]]
        self.weights = [np.random.randn(nex, prev)
                        for nex, prev in zip(self.sizes[1:], self.sizes[:-1])]


    def weight_decay(self, w, lmbda, eta, n):
        return self._weight_decay(w, lmbda, eta, n)            
        
    
    def feedforward(self, a):
        '''Returns the network output when "a" is the input'''
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(w @ a + b)
        return a


    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda, test_data = None):
        '''
        Trains the network using stochastic gradient descent using minibatches
        by employing the backpropagation algorithm. 

        training_data   - list of touples (x,y), representing inputs and desired outputs
        test_data       - same format as training_data
        epochs          - number of times the training set is split into minibatches
        mini_batch_size - size of minibatches
        eta             - learning rate
        '''
        if test_data:
            n_test = len(test_data)
            best_classified = 0

        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] 
                            for k in range(0,n,mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)
        
            if test_data:
                correct = self.evaluate(test_data)
                print("Epoch {0}: {1} / {2}".format(epoch+1, correct, n_test))
                if correct > best_classified:
                    best_classified = correct
                    self.score = best_classified/n_test
                    self.best_biases = self.biases
                    self.best_weights = self.weights
            else:
                print("Epoch {0} complete".format(epoch+1))
        

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            # del C_Xj /del b  &  del C_Xj /del w
            delta_grad_b, delta_grad_w = self.backprop(x,y)

            # grad C = 1/m *Sum(grad C_Xj) -> apply to b,w separately, w/o 1/m
            grad_b = [gb + dgb for gb, dgb in zip(grad_b, delta_grad_b)]
            grad_w = [gw + dgw for gw, dgw in zip(grad_w, delta_grad_w)]

        # update b,w: (b/w)' = (b/w) - eta *1/m *grad C (wrt. b/w)
        self.biases = [b - eta/len(mini_batch)*gb for b, gb in zip(self.biases, grad_b)]
        self.weights = [self.weight_decay(w, lmbda, eta, n) 
                        - eta/len(mini_batch)*gw for w, gw in zip(self.weights, grad_w)]
    

    def backprop(self, x, y):
        '''
        Returns a touple (grad_b, grad_w), describing the gradient of the cost function
        C_x (wrt. one training image). grad_b and grad_w are lists of np-arrays of same
        shape as self.biases and self.weights. x and y describe one training datum.
        '''
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations  = [x]
        zs = []

        # feedforward process: run the network on x and save all weighted inputs (zs)
        # and activations
        for b,w in zip(self.biases, self.weights):
            z = w @ activation + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # errors on last layer and from that the partial derivatives
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        grad_b[-1] = delta
        grad_w[-1] = delta @ np.transpose(activations[-2])

        # propagation of the error to previous layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.multiply(np.transpose(self.weights[-l+1]) @ delta, sigmoid_prime(z))
            grad_b[-l] = delta
            grad_w[-l] = delta @ np.transpose(activations[-l-1])

        return (grad_b, grad_w)
        

    def evaluate(self, test_data):
        '''Returns the number of correctly classified test datums'''
        predictions = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        correct = sum(int(aL == y) for (aL, y) in predictions)
        return correct
    

    def save_model(self, filename="models/mnist_model2.npz"):
        '''Saves biases and weights to a compressed .npz file.'''
        np.savez_compressed(filename,
            **{f"b{i}": b for i, b in enumerate(self.best_biases)},
            **{f"W{i}": w for i, w in enumerate(self.best_weights)})
        print(f"Model saved to {filename}")


    def load_model(self, filename="models/mnist_model2.npz"):
        '''
        Loads the biases and weights from a .npz file.
        Returns the biases and weights as lists of numpy arrays.
        '''
        data = np.load(filename)
        self.weights = [data[f"W{i}"] for i in range(int(len(data.keys())/2))]
        self.biases = [data[f"b{i}"] for i in range(int(len(data.keys())/2))]
        data.close()
        print(f"Model loaded from {filename}")

    

def sigmoid(z):
    '''Sigmoid function with input "z"'''
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    '''Derivative of the sigmoid function.'''
    return sigmoid(z)*(1-sigmoid(z))