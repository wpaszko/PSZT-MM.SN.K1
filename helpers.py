"""
Module with helpers function
"""

import copy
import numpy as np

from sklearn.model_selection import KFold


def cross_val_score(model, x, y, k=5, epochs=100, batch_size=1, shuffling=True):
    """
    Perform k-cross validation and return mean score

    Args:
        model: model to validate
        x: input
        y: target
        k: number of validations
        epochs: iterations of learning
        batch_size: samples per gradient update
        shuffling: whether to randomly shuffle data in each epoch

    Returns:
        mean score of k validations
    """
    scores = []
    fresh_model = copy.deepcopy(model)

    for train_idx, test_idx in KFold(k).split(x, y):
        model = fresh_model

        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.0, shuffling=shuffling)

        scores.append(model.score(x_test, y_test))

    return np.mean(scores)


def one_hot_encode(x):
    """
    Turns categories into ones and zeros in each sample [2], [3] -> [0, 0, 1, 0], [0, 0, 0, 1]
    Args:
        x: 1d array of categories of each sample

    Returns:
        2d array of one-hot encoded categories
    """
    a = x
    x = np.zeros((a.size, a.max() + 1))
    x[np.arange(a.size), a] = 1
    return x
