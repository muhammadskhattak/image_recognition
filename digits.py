""" Muhammad Khattak
    2018-04-12
    Version  1.0
    Code used from this website
    http://neuralnetworksanddeeplearning.com/chap1.html
"""
from typing import Tuple, List
import csv, random
import numpy as np

FILE_PATH :str = 'train.csv'

class Network:
    """ A neural network for machine learning."""

    def __init__(self, sizes: List[int]) -> None:
        """ Create a new network with layers of the specified size. """
        self.num_layers = len(sizes)
        self.sizes = sizes
        #Generate Gaussian distributions with mean 0 and standard deviation 1
        #The np function returns a matrix of dimensions y x 1
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        """ Apply the network to a to produce the expected output. """
        for b, w in zip(self.biases, self.weights):
            #Apply all the weights to the input a
            a_prime = sigmoid(np.dot(w, a) + b)
        return a_prime

    def stochastic_gradient_descent(self, training_data, mini_batch_size: int,
                                    epochs: int, eta: float, test_data =None):
        """ Apply gradient descenet to a subset of the training data so that
        this network can 'learn'. If test_data is provided, the algorithm will
        test against it after each epoch. """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_bathces = [training_data[k:k+mini_batch_size]
                            for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_bathces:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {}: {} / {}".format(j, self.evaluate(test_data),
                                                 n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta: float) -> None:
        """ Update the networks weights and biases by applying gradient descent
        on the mini_batch. eta is the learning rate."""
        #Initialize a new array with all zero of the corresponding dimensions b
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        #Initialize a new array with all zero of the corresponding dimensions w
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_new_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #Applying the gradient descent algorithm
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """ Return a tuple ``(nabla_b, nabla_w)`` representing the gradient
        for the cost function C_x. ``nabla_b`` and ``nabla_w`` are
        layer-by-layer lists of numpy arrays. """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedfoward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        delta = self.cost_derivate(activation[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """ Return the amount of test inputs which the neural net outputs the
        correct result.
        """
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum([int(x == y) for (x, y) in test_results])

    def cost_derivate(self, output_activations, y):
        """ Return the vector of partial derivatives (\partial C_x / \partial a)
         for the output activations.
        """
        return (output_activations - y)

def sigmoid(z: float) -> float:
    """ Apply the sigmoid function to z"""
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z: float) -> float:
    """ Apply the derivaitve of the sigmoid function to z."""
    return sigmoid(z) * (1 - sigmoid(z))

def parse_data() -> List[Tuple[int, List[int]]]:
    """ Parse the data in the csv file into labels and the pixels."""
    data = open(FILE_PATH)
    reader = csv.reader(data)
    training_set = []
    for row in reader:
        label = int(row[0])
        pixels = [feature_scale(int(pixel)) for pixel in row[1:]]
        training_set.append((label, pixels))
    data.close()
    return training_set

def feature_scale(value: int) -> float:
    """ Normalize the feature value from [0, 255] to [0, 1] using the formula
    for nomalizing: value := value - min(value) / max(value) - min(value) """
    return value / 255

if __name__ == '__main__':
    data = parse_data()
