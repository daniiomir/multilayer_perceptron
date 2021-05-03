import numpy as np
from src.layers import Layer


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        relu_forward = np.maximum(0, input)
        return relu_forward

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))

    def backward(self, input, grad_output):
        return 1 - self.forward(input) ** 2


class SoftMax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        e_x = np.exp(input - np.max(input))
        return e_x / e_x.sum(axis=0)

    def backward(self, input, grad_output):
        p = self.forward(input)
        return p * (1 - p)
