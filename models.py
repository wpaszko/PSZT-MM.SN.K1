"""
Module with models of neural networks

Usage:
m = model(...)
m.compile(...)

m.fit(...)
m.evaluate(...)

m.predict(...)
"""

from abc import ABC, abstractmethod
import layers as lrs


class Model(ABC):
    """
    Model base class

    Attributes:
        layers: list of neuron layers
        loss: loss function
    """

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    @abstractmethod
    def fit(self, x, y, epochs=1, batch_size=1, validation_split=0.0, shuffle=True):
        """
        Train model

        Args:
            x: input
            y: target
            epochs: number of times to iterate over entire input
            batch_size: number of samples per gradient update
            validation_split: what portion of samples to use for validation
            shuffle: whether to shuffle data before each epoch

        Returns:
            history of loss values throughout training process
        """
        pass

    @abstractmethod
    def evaluate(self, x, y):
        """
        Measure loss on given input

        Args:
            x: input
            y: target

        Returns:
            loss value
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        Predict output of given input

        Args:
            x: input

        Returns:
            predicted output
        """
        pass


class Sequential(Model):
    """
    Simplest neural network model with layer to layer connection
    """

    def fit(self, x, y, epochs=1, batch_size=1, validation_split=0.0, shuffle=True):
        pass

    def evaluate(self, x, y):
        pass

    def predict(self, x):
        pass


class MultiLayerPerceptron(Sequential):
    """
    Multi-layer perceptron
    """

    def __init__(self, input_size, output_size, hidden_layers_num, hidden_layers_sizes, loss):
        """
        Create multi-layer perceptron

        Args:
            input_size: neurons on input layer
            output_size: neurons on output layer
            hidden_layers_num: number of hidden layers
            hidden_layers_sizes: list of neurons on each hidden layer
            loss:
        """
        layers = None  # TODO: create layers
        super().__init__(layers, loss)
