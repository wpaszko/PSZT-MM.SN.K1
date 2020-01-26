import layers
import numpy as np

layers = [layers.Dense(4, 3),
          layers.ReLU(),
          layers.Dense(3, 2),
          layers.Softmax()]

input = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])

result = input

for layer in layers:
    result = layer.forward(result)

result = result / 2

for layer in reversed(layers):
    result = layer.backward(result)

print(result)

result = input

for layer in layers:
    result = layer.forward(result)

result = result / 2

for layer in reversed(layers):
    result = layer.backward(result)

print(result)
