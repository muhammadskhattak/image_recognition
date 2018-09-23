""" Muhammad Khattak
    2018-04-12
    Version  1.0
"""
from typing import Tuple, List
from vector import Vector
import csv, random, math
import numpy as np

class Network:
    def __init__(self, sizes: List[int]) -> None:
        """ Create a new network with layers of the specified size."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [Vector([0 for i in range(j)]) for j in sizes[1:]]
        self.weights = [Vector([0 for i in range(j)]) for j in sizes[:-1]]

    

def sigmoid(z: float) -> float:
    """ Apply the sigmoid function to z. """
    return 1.0 / (1.0 + math.exp(-z))

def parse_data() -> List[Tuple[int, List[int]]]:
    """ Parse the data in the csv file into labels and the pixels."""
    data = open(FILE_PATH)
    reader = csv.reader(data)
    training_set = []
    for row in reader:
        label = int(row[0])
        pixels = [feature_scale(int(pixel)) for pixel in row[1:]]
        training_set.append((label, Vector(pixels)))
    data.close()
    return training_set

def feature_scale(value: int) -> float:
    """ Normalize the feature value from [0, 255] to [0, 1] using the formula
    for nomalizing: value := value - min(value) / max(value) - min(value) """
    return value / 255


