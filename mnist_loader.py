'''
mnist_loader.py
~~~~~~~~~~~~~~~

A module to load the MNIST dataset as lists of touples (training_data, test_data),
where both are list of touples [(x,y)], where x are numpy arrays of shape 28*28=784x1
and y are numpy arrays of shape 10x1, where the corresponding lable is set to 1,
the rest to 0.
'''

#### Libraries
# Third-party libraries
from sklearn.datasets import fetch_openml
import numpy as np

def load_data(train_split, val_split):
    # load data
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    # assign data
    X, Y = mnist.data, mnist.target

    # normalize inputs
    X = X / 255.0
    
    # output as integers and reshape
    Y = Y.astype(int)
    Y = [vectorize_results(y) for y in Y]

    # generate list of touples and split it
    n = np.shape(X)[0]
    all_data = [(x.reshape(-1, 1), y) for x, y in zip(X, Y)]
    training_data = all_data[:train_split]
    validation_data = all_data[train_split:train_split+val_split]
    test_data = all_data[train_split+val_split:]

    return(training_data, validation_data, test_data)



def vectorize_results(j):
    '''turns j into the j-th canonical vector.'''
    e = np.zeros((10,1))
    e[j] = 1
    return e