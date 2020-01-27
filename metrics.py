"""
Module with metrics helpers
"""

from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """
    Loss base class
    """

    @abstractmethod
    def loss(self, y_true, y_pred):
        """
        Calculate loss across batch

        Args:
            y_true: true labels
            y_pred: predicted labels

        Returns:
            loss value
        """
        pass

    @abstractmethod
    def gradient(self, y_true, y_pred):
        """
        Calculate gradient of loss with respect to y_pred

        Args:
            y_true: true labels
            y_pred: predicted labels

        Returns:
            idk yet
        """
        pass


class CrossEntropyLoss(Loss):

    epsilon = 1e-5

    def loss(self, y_true, y_pred):
        """
        Computes cross entropy between true labels and predictions
        """

        predictions = np.clip(y_pred, self.epsilon, 1. - self.epsilon)
        batch_size = predictions.shape[0]
        ce = -np.sum(y_true * np.log(predictions)) / batch_size
        return ce

    def gradient(self, y_true, y_pred):
        """
        Gradient of cross entropy with relation to y_pred
        """
        predictions = np.clip(y_pred, self.epsilon, 1. - self.epsilon)
        return - y_true / predictions


class Score(ABC):
    """
    Score base class
    """

    @abstractmethod
    def score(self, y_true, y_pred):
        """
        Calculates predictions score
        Args:
            y_true: one-hot encoded target
            y_pred: one-hot encoded predictions

        Returns:
            score
        """
        pass


class AccuracyScore(Score):
    def score(self, y_true, y_pred):
        """
        Calculates score as fraction of matches
        Args:
            y_true: one-hot encoded target
            y_pred: one-hot encoded predictions

        Returns:
            score
        """

        true_classes = np.argmax(y_true, axis=1)
        predicted_classes = np.argmax(y_pred, axis=1)

        return np.mean(true_classes == predicted_classes)
