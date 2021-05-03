import numpy as np


class Layer:
    def __init__(self):
        self.require_grad = False

    def forward(self, input):
        raise NotImplementedError

    def backward(self, input, grad_output):
        """
        Performs a backpropagation step through the layer, with respect to the given input.
        To compute loss gradients w.r.t input, we need to apply chain rule (backprop):
        d loss / dx  = (d loss / d layer) * (d layer / dx)
        Luckily, we already receive d loss / d layer as input, so you only need to multiply it by d layer / d x.
        If our layer has parameters (e.g. dense layer), we also need to update them here using d loss / d layer
        The gradient of a dummy layer is precisely grad_output, but we'll write it more explicitly
        """
        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input)


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        super().__init__()
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0,
                                        scale=np.sqrt(2 / (input_units + output_units)),
                                        size=(input_units, output_units))
        self.biases = np.zeros(output_units)
        self.require_grad = True
        self.weights_grad = None
        self.biases_grad = None

    def forward(self, input):
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        """
        Compute df / dx = df / d dense * d dense / dx
        Where d dense / dx = weights transposed
        """
        grad_input = np.dot(grad_output, self.weights.T)

        self.weights_grad = np.dot(input.T, grad_output)
        self.biases_grad = grad_output.mean(axis=0) * input.shape[0]

        assert self.weights_grad.shape == self.weights.shape and self.biases_grad.shape == self.biases.shape

        # self.weights -= self.learning_rate * grad_weights
        # self.biases -= self.learning_rate * grad_biases
        return grad_input
