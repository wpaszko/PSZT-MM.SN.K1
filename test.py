import argparse
import models
import helpers

from sklearn.datasets import load_iris


def learn_and_evaluate(neurons, learn_rate, k, epochs, batch_size):
    database = load_iris()

    x = database.data
    y = database.target

    input = x
    target = helpers.one_hot_encode(y)

    mlp = models.MultiLayerPerceptron(input.shape[1], target.shape[1], neurons, learn_rate=learn_rate)

    return helpers.cross_val_score(mlp, input, target, k, epochs, batch_size)


def parse_args(parser):
    args = parser.parse_args()

    for n in args.neurons:
        if not n >= 1:
            parser.error("Neuron numbers should be integers > 1")
    if not (0.0 <= args.l <= 1.0):
        parser.error("Learning rate should be between 0.0 and 1.0")
    if not args.k >= 1:
        parser.error("K-cross validation size should be at least one")
    if not args.e >= 1:
        parser.error("There has to be at least one epoch")
    if not args.b >= 0:
        parser.error("Batch size has to be >= 0")

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="neural network classifying iris flowers")
    parser.add_argument('neurons', type=int, nargs='+', help='number of neurons on hidden layers (list)')
    parser.add_argument('-l', type=float, default=0.01, help='learning rate')
    parser.add_argument('-k', type=int, default=5, help='k-cross validation size')
    parser.add_argument('-e', type=int, default=100, help='epochs')
    parser.add_argument('-b', type=int, default=1, help='batch size (0 - full input)')
    args = parse_args(parser)

    loss = learn_and_evaluate(args.neurons, args.l, args.k, args.e, args.b)
    print("Loss value:", loss)
