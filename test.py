import models
import numpy as np

input = np.array([[1, 2, 3, 4]])
target = np.array([0, 1])

mlp = models.MultiLayerPerceptron(input.shape[1], target.shape[1], [4, 3])

history = mlp.fit(input, target, epochs=100)
