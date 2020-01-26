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
        bias: constant value of 1
        forward_input: remembered input for forward pass, so we don't have to pass it explicitly from outside
    """

    def __init__(self):
        self.bias = 1
        self.forward_input = None

    @abstractmethod
    def forward(self, input):
        """
        Process input the way particular layer does

        Args:
            input: numpy array of shape (1, size) (WARNING: it has to be 2d, shape (size,) is invalid)

        Returns:
            processed input
        """

        self.forward_input = input

    @abstractmethod
    def backward(self, grad_output):
        """
        Update layer params with back-propagating gradient
        Args:
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
        super().__init__()
        norm_center = 0
        norm_std = 2 / math.sqrt(input_size + output_size)  # weight should be close to zero, depending on input size
        norm_trunk = 2 * norm_std  # cut off everything after 2 standard deviations

        norm_dist = stats.truncnorm((-(norm_trunk - norm_center)) / norm_std, (norm_trunk - norm_center) / norm_std, loc=0, scale=norm_std)

        self.weights = norm_dist.rvs(size=(input_size, output_size))
        self.bias_weights = norm_dist.rvs(size=output_size)

        self.learn_rate = learn_rate

    def forward(self, input):
        """
        Compute dot product of input and weights and add bias
        """
        super().forward(input)

        return np.dot(input, self.weights) + self.bias_weights * self.bias

    def backward(self, grad_output):
        """
        Compute df/dw given grad from next layer (which should be activation layer) df/da
        according to df/dw = df/da * da/dw

        da/dw is just weights
        """

        grad = np.dot(grad_output, self.weights.T)  # compute df/dw = df/da * da/dw

        grad_weights = np.dot(self.forward_input.T, grad_output)  # compute gradient with respect to weights df/dw * w
        self.weights = self.weights - self.learn_rate * grad_weights  # update weights with computed gradient

        grad_biases = grad_output.mean(axis=0) * self.forward_input.shape[0]  # since biases don't depend on input, average gradient across batch
        self.bias_weights = self.bias_weights - self.learn_rate * grad_biases    # update weights with computed gradient (we use grad_weights because df/db = df/dw)

        return grad


class ReLU(Layer):
    """
    ReLu activation layer
    """

    def forward(self, input):
        """
        ReLU activation is linear, capped at 0: max(0, input)
        """
        super().forward(input)

        return np.maximum(0, input)

    def backward(self, grad_output):
        """
        Derivative of ReLU is 0 for input < 0, and 1 for input > 0
        """

        return grad_output * (self.forward_input > 0)


class LeakyReLU(Layer):
    """
    LeakyReLu activation layer

    Attributes:
        alpha: factor of scaling down negative input
    """

    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        """
        ReLU activation is linear, but instead of capping at 0, we scale negative values down: max(alpha * input, input)
        """
        super().forward(input)

        return np.maximum(self.alpha * input, input)

    def backward(self, grad_output):
        """
        Derivative of LeakyReLU is alpha for input < alpha, and 1 for input > 0
        """

        da = np.where(self.forward_input > 0, 1, self.alpha)

        return grad_output * da


class ELU(Layer):
    """
    ELU activation layer

    Attributes:
        alpha: factor of scaling down negative input
    """

    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        """
        ELU is like LeakyReLU, but instead of scaling linearly, we scale exponentially: alpha * (exp(input) - 1)
        """
        super().forward(input)

        return np.maximum(self.alpha * (np.exp(input) - 1), input)

    def backward(self, grad_output):
        """
        Derivative of LeakyReLU is alpha for input < alpha, and 1 for input > 0
        """

        da = np.where(self.forward_input > 0, 1, self.alpha * np.exp(self.forward_input))

        return grad_output * da


class Softmax(Layer):
    """
    Softmax activation layer

    Attributes:
        forward_pass: remembered value from forward passing to not calculate it twice
    """

    def __init__(self):
        super().__init__()
        self.forward_pass = None

    def forward(self, input):
        """
        Softmax turns each input value into probability scaled through whole input so [2, 1, 0] -> [0.7, 0.2, 0.1]
        """
        shift = input - np.max(input)  # shifting all values towards negative to get rid of large values for better numerical stability
        exps = np.exp(shift)
        self.forward_pass = exps / np.sum(exps)
        return self.forward_pass

    def backward(self, grad_output):
        """
        Softmax derivative (kinda complicated)
        """

        ds = np.array([np.diagflat(fp) - np.outer(fp, fp.T) for fp in self.forward_pass])

        return np.array([np.dot(grad_output[i], ds[i]) for i in range(grad_output.shape[0])])
