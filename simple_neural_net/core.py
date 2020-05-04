
class NeuralNet:
    def __init__(
        self,
        layers,
        cost,
    ):
        self.layers = layers
        self.cost = cost
        self.num_layers = len(layers)

    def train(
        self,
        num_iterations,
        learning_rate,
        ground_truth_inputs,
        ground_truth_outputs
    ):
        for iteration_index in num_layers:
            inputs = ground_truth_inputs
            for layer_index in range(num_layers):
                outputs = layers[layer_index].forward_propagate(inputs)
                inputs = outputs
            self.cost = cost.cost(outputs, ground_truth_outputs)
            d_cost_d_outputs = cost.d_cost_d_outputs(outputs, ground_truth)
            for layer_index in reversed(range(num_layers)):
                d_cost_d_inputs = layers[layer_index].backward_propagate(d_cost_d_outputs)
                d_cost_d_outputs = d_cost_d_inputs
