import models
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import datasets

database = load_iris()

X = database.data
y = database.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#one hot encoding on y_train array
a = y_train
y_train = np.zeros((a.size, a.max()+1))
y_train[np.arange(a.size),a] = 1

input = X_train
target = y_train

mlp = models.MultiLayerPerceptron(input.shape[1], target.shape[1], [4, 3])

history = mlp.fit(input, target, epochs=100)

print(history)

predictions = mlp.predict(X_test)

print(predictions)