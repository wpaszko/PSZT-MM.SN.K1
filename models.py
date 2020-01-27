"""
Module with models of neural networks

Usage:
m = model(...)
m.compile(...)

m.fit(...)
m.evaluate(...)

m.predict(...)
"""

import layers as lrs
import metrics as mtr
import numpy as np

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Model(ABC):
    """
    Model base class

    Attributes:
        layers: list of neuron layers
        loss: Loss object
        scorer: Score object
    """

    def __init__(self, layers, loss, scorer):
        self.layers = layers
        self.loss = loss
        self.scorer = scorer

    @abstractmethod
    def fit(self, x, y, epochs=1, batch_size=1, validation_split=0.0, shuffling=True):
        """
        Train model

        Args:
            x: input
            y: target
            epochs: number of times to iterate over entire input
            batch_size: number of samples per gradient update
            validation_split: what portion of samples to use for validation
            shuffling: whether to shuffle data before each epoch

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

    @abstractmethod
    def score(self, x, y):
        """
        Measure model score on given input

        Args:
            x: input
            y: target

        Returns:
            score value
        """
        pass


class Sequential(Model):
    """
    Simplest neural network model with layer to layer connection
    """

    def fit(self, x, y, epochs=1, batch_size=1, validation_split=0.1, shuffling=True):
        history = {'loss': [], 'val_loss': [], 'score': []}  # history of losses

        if batch_size == 0:
            batch_size = x.shape[0]

        if 0.0 < validation_split < 1.0:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split)  # split data into training and validation sets
        else:
            x_train, x_val, y_train, y_val = x, np.array([]), y, np.array([])

        for i in range(epochs):  # learn epochs iterations

            x_e = x_train
            y_e = y_train

            if shuffling:
                x_e, y_e = shuffle(x_train, y_train)  # randomly shuffle training data in each epoch

            # divide data into batches
            batches = [x_e[i:i + batch_size] for i in range(0, len(x_e), batch_size)]
            target_batches = [y_e[i:i + batch_size] for i in range(0, len(y_e), batch_size)]
            batch_losses = []

            for j, batch in enumerate(batches):
                predictions = self.predict(batch)  # forward pass

                batch_loss = self.loss.loss(target_batches[j], predictions)  # loss
                batch_losses.append(batch_loss)

                gradient = self.loss.gradient(target_batches[j], predictions)  # gradient of loss function

                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient)  # backward pass

            # after all batches pass losses into history
            mean_batch_loss = np.mean(batch_losses)
            history['loss'].append(mean_batch_loss)
            if x_val.size != 0:  # extra data: loss and score on validation data (if present)
                history['val_loss'].append(self.evaluate(x_val, y_val))
                history['score'].append(self.score(x_val, y_val))

        return history

    def evaluate(self, x, y):
        return self.loss.loss(y, self.predict(x))

    def predict(self, x):
        input = x

        for layer in self.layers:
            input = layer.forward(input)

        return input

    def score(self, x, y):
        return self.scorer.score(y, self.predict(x))


class MultiLayerPerceptron(Sequential):
    """
    Multi-layer perceptron
    """

    def __init__(self, input_size, output_size, hidden_layers_sizes, loss=mtr.CrossEntropyLoss(), learn_rate=0.01, problem='classification', scorer=mtr.AccuracyScore()):
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
            layers.append(lrs.LeakyReLU())

        layers.append(lrs.Dense(layer_sizes[-2], layer_sizes[-1], learn_rate))

        if problem == 'classification':
            layers.append(lrs.Softmax())

        super().__init__(layers, loss, scorer)
