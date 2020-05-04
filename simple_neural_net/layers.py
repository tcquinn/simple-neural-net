import numpy as np

class Layer:
    def __init__(
        num_inputs,
        num_outputs,
        activation,
        weight_initialization_multiplier = 0.01,
        bias_initialization_multiplier = 0.0
    ):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation = activation
        self.weight_initialization_multiplier = weight_initialization_multiplier
        self.bias_initialization_multiplier = bias_initialization_multiplier
        self.weights = np.multiply(
            np.random.randn(self.num_outputs, self.num_inputs),
            self.weight_initialization_multiplier
        )
        self.biases = np.multiply(
            np.random.randn(self.num_outputs, 1),
            self.bias_initialization_multiplier
        )

    def forward_propagate(inputs):
        # inputs: (num_inputs, num_examples)
        # Z: (num_outputs, num_examples)
        # outputs: (num_outputs, num_examples)
        self.inputs = inputs # (num_inputs, num_examples)
        self.Z = np.dot(self.weights, self.inputs) + self.biases # (n_outputs, num_examples)
        self.outputs = self.activation.activation(self.Z) # (num_outputs, num_examples)
        return outputs

    def backward_propagate(d_cost_d_outputs):
        # d_cost_d_outputs: (num_outputs, num_examples)
        # d_cost_d_Z: (num_outputs, num_examples)
        # d_cost_d_weights (num_outputs, num_inputs)
        # d_cost_d_biases: (n_outputs, 1)
        # d_cost_d_inputs: (n_inputs, num_training_examples)
        self.d_cost_d_outputs = d_cost_d_outputs
        self.d_cost_d_Z = self.d_cost_d_outputs * self.activation.d_activation_d_Z(self.Z, self.outputs) # (num_outputs, num_examples)
        self.d_cost_d_weights = np.dot(self.d_cost_d_Z, self.inputs.T) # (n_outputs, n_inputs)
        self.d_cost_d_biases = np.sum(d_cost_d_Z, axis=1, keepdims=True) # (n_outputs, 1)
        self.d_cost_d_inputs = np.dot(self.weights.T, self.d_cost_d_Z) # (n_inputs, num_training_examples)
        return d_cost_d_inputs
