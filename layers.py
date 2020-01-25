"""
Module with layers of neural networks

Usage:
layer.forward(...)
layer.backward(...)
"""

from abc import ABC, abstractmethod
import numpy as np
import math
from scipy import stats


class Layer(ABC):
    """
    Layer base class

    Attributes:
        n: number of neurons
    """

    bias = 1

    def __init__(self, neurons):
        self.n = neurons

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


class Dense(Layer):
    """
    Dense layer - every neuron in previous layer connected to every neuron in this layer

    Attributes:
        weights: array of shape (input size, output size) containing current weights of connections between neurons and input
        bias_weights: array of shape (output size) containing weights of connections between neurons and bias (constant input 1)
    """

    def __init__(self, input_size, output_size, learn_rate=0.01):
        super().__init__(output_size)

        norm_center = 0
        norm_std = 2 / math.sqrt(input_size)  # weight should be close to zero, depending on input size
        norm_trunk = 2 * norm_std  # cut off everything after 2 standard deviations

        norm_dist = stats.truncnorm((-(norm_trunk - norm_center)) / norm_std, (norm_trunk - norm_center) / norm_std, loc=0, scale=norm_std)

        self.weights = norm_dist.rvs(size=(input_size, output_size))
        self.bias_weights = norm_dist.rvs(size=output_size)

        self.learn_rate = learn_rate

    def forward(self, input):
        """
        Compute dot product of input and weights and add bias
        """

        return np.dot(input, self.weights) + self.bias_weights * self.bias

    def backward(self, input, grad_output):
        """
        Compute df/dw given grad from next layer (which should be activation layer) df/da
        according to df/dw = df/da * da/dw

        da/dw is just weights
        """

        grad = np.dot(grad_output, self.weights.T)  # compute df/dw = df/da * da/dw

        grad_weights = np.dot(input.T, grad)  # compute gradient with respect to weights df/dw * w
        self.weights = self.weights - self.learn_rate * grad_weights  # update weights with computed gradient
        self.bias_weights = self.bias_weights - self.learn_rate * grad_weights    # update weights with computed gradient (we use grad_weights because df/db = df/dw)

        return grad


class ReLU(Layer):
    """
    ReLu activation layer
    """

    def forward(self, input):
        """
        ReLU activation is linear, capped at 0: max(0, input)
        """

        return np.maximum(0, input)

    def backward(self, input, grad_output):
        """
        Derivative of ReLU is 0 for input < 0, and 1 for input > 0
        """

        return grad_output * (input > 0)
