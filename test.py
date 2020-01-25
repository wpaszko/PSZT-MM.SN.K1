import models

input = [1]
target = [1]

input_size = 1
output_size = 1
neuron_sizes = [1, 1, 1]


def loss_function(output_value, target_value):
    return 0


mlp = models.MultiLayerPerceptron(input_size, output_size, len(neuron_sizes), neuron_sizes, loss_function)

mlp.fit(input, target)

print("Prediced value: ", mlp.predict(input[0]))
