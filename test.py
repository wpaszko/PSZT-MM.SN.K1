import argparse
import models
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def learn_and_evaluate(neurons, learn_rate, epochs, batch_size, test_size):
    database = load_iris()

    x = database.data
    y = database.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    # one hot encoding on y_train array
    a = y_train
    y_train = np.zeros((a.size, a.max() + 1))
    y_train[np.arange(a.size), a] = 1

    input = x_train
    target = y_train

    mlp = models.MultiLayerPerceptron(input.shape[1], target.shape[1], neurons, learn_rate=learn_rate)

    mlp.fit(input, target, epochs=epochs, batch_size=batch_size)

    a = y_test
    y_test = np.zeros((a.size, a.max() + 1))
    y_test[np.arange(a.size), a] = 1

    return mlp.evaluate(x_test, y_test)


def parse_args(parser):
    args = parser.parse_args()

    for n in args.neurons:
        if not n >= 1:
            parser.error("Neuron numbers should be integers > 1")
    if not (0.0 <= args.l <= 1.0):
        parser.error("Learning rate should be between 0.0 and 1.0")
    if not args.e >= 1:
        parser.error("There has to be at least one epoch")
    if not args.b >= 0:
        parser.error("Batch size has to be >= 0")
    if not (0.0 <= args.t <= 1.0):
        parser.error("Test set size should be between 0.0 and 1.0")

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="neural network classifying iris flowers")
    parser.add_argument('neurons', type=int, nargs='+', help='number of neurons on hidden layers (list)')
    parser.add_argument('-l', type=float, default=0.01, help='learning rate')
    parser.add_argument('-e', type=int, default=100, help='epochs')
    parser.add_argument('-b', type=int, default=1, help='batch size (0 - full input)')
    parser.add_argument('-t', type=float, default=0.2, help='test set size')
    args = parse_args(parser)

    loss = learn_and_evaluate(args.neurons, args.l, args.e, args.b, args.t)
    print("Loss value:", loss)
