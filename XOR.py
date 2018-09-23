""" Muhammad Khattak
    2018-04-13
    Version 1.0
"""
from typing import List, Tuple, Union
import numpy as np
import random

TRAINING_SET = [(0, np.array([0, 0])), (1, np.array([0, 1])), (1, np.array([1, 0])), (0, np.array([1, 1]))]

class Network:
    """ A neural network for learning XOR. """
    sizes: List[int]
    num_layers: int

    def __init__(self, sizes: List[int]) -> None:
        """ Create a new network with layers of the specified sizes."""
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [np.random.randn(y, x) for y, x in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(1, y) for y in sizes[1:]]

    def feed_forward(self, inp):
        """ Apply the weights and biases to the input vector. """
        for w in self.weights:
            inp = sigmoid(np.dot(inp, w))
        res = 0.5
        if inp[0] > 0.9:
            res = 1
        elif inp[0] < 0.1:
            res = 0
        return res

        return round(inp[0])

    def gradient_descent(self, inp, expected, eta: float) -> None:
        """ Apply gradient descent to the network. """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        delta_w = self.backprop(inp, expected)
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_w)]
        self.weights = [w - (eta) * nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, inp, expected):
        """ Change the weights of this neural net accordingly. """
        new_weights = [np.zeros(w.shape) for w in self.weights]
        #Forward propagate, keeping track of each states sum:
        sum_layer = np.array([])
        sum_layers = []
        activation = inp
        activations = [inp]
        for w in self.weights:
            sum_layer = np.dot(activation, w)
            sum_layers.append(sum_layer)
            activation = sigmoid(sum_layer)
            activations.append(activation)
        #Now backpropagate
        delta = sigmoid_prime(sum_layers[-1]) * (activations[-1] - expected)
        new_weights[-1] = np.dot(delta, activations[-2][np.newaxis])[np.newaxis].transpose()
        sp = sigmoid_prime(sum_layers[-2])
        delta = np.dot(self.weights[-1].transpose(), delta[0]) * sp
        new_weights[-2] = np.dot(activations[-3].transpose()[np.newaxis].transpose(), delta)
        return new_weights

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def test(network: Network, epoch, eta, num_training, num_tests):
    for e in range(epoch):
        print('--Training--')
        for i in range(num_training):
            training_data = random.choice(TRAINING_SET)
            net.gradient_descent(training_data[1], training_data[0], eta)
        print('--Testing--')
        correct = 0
        for i in range(num_tests):
            testing_data = random.choice(TRAINING_SET)
            if net.feed_forward(testing_data[1]) == testing_data[0]:
                correct += 1
        print('--Evaluation--')
        print((correct / num_tests) * 100)


if __name__ == '__main__':
    net = Network([2,4,1])
    test(net, 25, 0.05, 10000, 1000)
    import os
    os.system('cls')
    txt = 'a'
    while txt is not '':
        txt = input('Input Logic (a, b):')
        if txt != '':
            txt = txt.split()
            txt = np.array([int(a[0]) for a in txt])
            print(net.feed_forward(txt))
