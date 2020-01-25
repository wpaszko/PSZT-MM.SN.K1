"""
Module with layers of neural networks

Usage:
layer.forward(...)
layer.backward(...)
"""

from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    """
    Layer base class

    Attributes:
        weights: array of shape (input size, output size) containing current weights of connections between neurons and input
        bias_weights: array of shape (output size) containg weights of connections between neurons and bias (constant input 1)
    """

    def __init__(self, input_size, output_size):
        # TODO: set random weights
        self.weights = np.zeros((input_size, output_size))
        self.bias_weights = np.zeros(output_size)

    @abstractmethod
    def forward(self, input):
        """
        Process input the way particular layer does

        Args:
            input: input

        Returns:
            processed input
        """
        pass

    @abstractmethod
    def backward(self, input, grad_output):
        """
        Update layer params with back-propagating gradient
        Args:
            input: input
            grad_output: gradient from next layer

        Returns:
            gradient of this layers with respect to given gradient and input
        """
        pass
