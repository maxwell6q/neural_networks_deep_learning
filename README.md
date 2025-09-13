# Neural Networks & Deep Learning - from scratch
This repository contains my personal implementation of a neural network from scratch in Python, based on the free online book [*Neural Networks and Deep Learning* by Michael Nielsen](http://neuralnetworksanddeeplearning.com/).

The goal of this project is **educational**: to understand the inner workings of neural networks without relying on deep learning frameworks such as TensorFlow or PyTorch.

## Current Contents
- `mnist_loader.py` contains functions to load the MNIST dataset
- `network.py` implements the actual network by the "Network" class
- `test_network.py` loads the dataset and trains the network


## Installation and Usage

Clone the repository:
```bash
git https://github.com/maxwell6q/neural_networks_deep_learning.git
cd neural_networks_deep_learning
```

Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

To train the model, run the `test_network.py` script, optionally adding different parameters.
