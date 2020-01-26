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
import metrics as mtr
import numpy as np


class Model(ABC):
    """
    Model base class

    Attributes:
        layers: list of neuron layers
        loss: Loss object
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
        history = []

        batches = [x[i:i + batch_size] for i in range(0, len(x), batch_size)]
        target_batches = [y[i:i + batch_size] for i in range(0, len(y), batch_size)]

        for i in range(epochs):

            batch_losses = []

            for j, batch in enumerate(batches):
                input = batch

                for layer in self.layers:
                    input = layer.forward(input)

                batch_loss = self.loss.loss(target_batches[j], input)
                batch_losses.append(batch_loss)

                gradient = self.loss.gradient(target_batches[j], input)

                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient)

            mean_batch_loss = np.mean(batch_losses)
            history.append(mean_batch_loss)

        return history

    def evaluate(self, x, y):
        pass

    def predict(self, x):
        pass


class MultiLayerPerceptron(Sequential):
    """
    Multi-layer perceptron
    """

    def __init__(self, input_size, output_size, hidden_layers_sizes, loss=mtr.CrossEntropyLoss(), learn_rate=0.01, problem='classification'):
        """
        Create multi-layer perceptron

        Args:
            input_size: neurons on input layer
            output_size: neurons on output layer
            hidden_layers_sizes: list of neurons on each hidden layer
            problem: problem to solve ('classification" or 'regression')
        """
        layer_sizes = [input_size] + hidden_layers_sizes + [output_size]
        layers = []

        for i in range(len(layer_sizes) - 2):
            layers.append(lrs.Dense(layer_sizes[i], layer_sizes[i + 1], learn_rate))
            layers.append(lrs.ReLU())

        layers.append(lrs.Dense(layer_sizes[-2], layer_sizes[-1], learn_rate))

        if problem == 'classification':
            layers.append(lrs.Softmax())

        super().__init__(layers, loss)
