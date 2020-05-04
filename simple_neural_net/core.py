
class NeuralNet:
    def __init__(
        self,
        layers,
        cost,
        prediction
    ):
        self.layers = layers
        self.cost = cost
        self.prediction = prediction
        self.num_layers = len(layers)

    def train(
        self,
        num_iterations,
        learning_rate,
        ground_truth_inputs,
        ground_truth_outputs
    ):
        for iteration_index in range(num_iterations):
            inputs = ground_truth_inputs
            for layer in self.layers:
                outputs = layer.forward_propagate(inputs)
                inputs = outputs
            cost = self.cost.cost(outputs, ground_truth_outputs)
            d_cost_d_outputs = self.cost.d_cost_d_outputs(outputs, ground_truth_outputs)
            for layer in reversed(self.layers):
                d_cost_d_inputs = layer.backward_propagate(d_cost_d_outputs)
                d_cost_d_outputs = d_cost_d_inputs
            for layer in self.layers:
                layer.weights += -learning_rate*layer.d_cost_d_weights
                layer.biases += -learning_rate*layer.d_cost_d_biases
            if iteration_index % 100 == 0:
                print('Iteration {}: cost {}'.format(
                    iteration_index,
                    cost
                ))

    def predict(
        self,
        inputs
    ):
        for layer_index in range(self.num_layers):
            outputs = self.layers[layer_index].forward_propagate(inputs)
            inputs = outputs
        return self.prediction.predictions(outputs)
