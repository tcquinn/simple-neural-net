
class NeuralNet:
    def __init__(
        self,
        layers,
        cost_model,
        prediction_model
    ):
        self.layers = layers
        self.cost_model = cost_model
        self.prediction_model = prediction_model
        self.num_layers = len(layers)

    def train(
        self,
        X,
        y,
        num_iterations,
        learning_rate
    ):
        for iteration_index in range(num_iterations):
            inputs = X
            for layer in self.layers:
                outputs = layer.forward_propagate(inputs)
                inputs = outputs
            cost = self.cost_model.cost(outputs, y)
            d_cost_d_outputs = self.cost_model.d_cost_d_outputs(outputs, y)
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
        X
    ):
        inputs = X
        for layer in self.layers:
            outputs = layer.forward_propagate(inputs)
            inputs = outputs
        y_hat = self.prediction_model.predictions(outputs)
        return y_hat
