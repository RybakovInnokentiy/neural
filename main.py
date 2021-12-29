import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class NeuralNetwork:

    def __int__(self):
        weights = np.array([2, 3])
        bias = 3

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, inputs_):
        out_h1 = self.h1.feedforward(inputs_)
        out_h2 = self.h2.feedforward(inputs_)
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1


network = NeuralNetwork()
inputs = np.array([4, 5])
print(network.feedforward(inputs))
