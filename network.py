'''
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.
'''

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np



class Network(object):

    def __init__(self, sizes):
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
        self.biases = [np.random.randn(nex, 1) for nex in sizes[1:]]
        self.weights = [np.random.randn(nex, prev) 
                        for nex, prev in zip(sizes[1:], sizes[:-1])]
        self.score = 0
        self.best_biases = [np.zeros((nex, 1)) for nex in sizes[1:]]
        self.best_weights = [np.zeros((nex, prev))
                             for nex, prev in zip(sizes[1:], sizes[:-1])]
        
    
    def feedforward(self, a):
        '''Returns the network output when "a" is the input'''
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(w @ a + b)
        return a


    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
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
                self.update_mini_batch(mini_batch, eta)
        
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
        

    def update_mini_batch(self, mini_batch, eta):
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
        self.weights = [w - eta/len(mini_batch)*gw for w, gw in zip(self.weights, grad_w)]
    

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
        delta = np.multiply(self.cost_derivative(activations[-1], y), sigmoid_prime(zs[-1]))
        grad_b[-1] = delta
        grad_w[-1] = delta @ np.transpose(activations[-2])

        # propagation of the error to previous layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.multiply(np.transpose(self.weights[-l+1]) @ delta, sigmoid_prime(z))
            grad_b[-l] = delta
            grad_w[-l] = delta @ np.transpose(activations[-l-1])

        return (grad_b, grad_w)
    

    def cost_derivative(self, output_activations, y):
        '''
        Computes the vector of partial derivatives del C_x /del a 
        where "a" is the vecotor of output activations
        '''
        return (output_activations-y) 
    

    def evaluate(self, test_data):
        '''Returns the number of correctly classified test datums'''
        predictions = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        correct = sum(int(aL == y) for (aL, y) in predictions)
        return correct
    

    def save_model(self, filename="mnist_model.npz"):
        '''Saves biases and weights to a compressed .npz file.'''
        np.savez_compressed(filename,
            **{f"b{i}": b for i, b in enumerate(self.best_biases)},
            **{f"W{i}": w for i, w in enumerate(self.best_weights)})
        print(f"Model saved to {filename}")


    def load_model(self, filename="mnist_model.npz"):
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