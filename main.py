import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:

    def __init__(self, weights, bias):
        self.weights = weights;
        self.bias = bias;

    def feedforwar(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias;
        return sigmoid(total);


weights = np.array([2, 1]);
bias = 5;
n = Neuron(weights, bias);

inputs = np.array([10, 12]);
print(n.feedforwar(inputs));
