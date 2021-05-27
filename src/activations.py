import numpy as np
from src.layers import Layer


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray, mode: str = 'train') -> np.ndarray:
        relu_forward = np.maximum(0, input)
        return relu_forward

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        relu_grad = input > 0
        return grad_output * relu_grad.astype(int)


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray, mode: str = 'train') -> np.ndarray:
        return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (1 - self.forward(input) ** 2)


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray, mode: str = 'train') -> np.ndarray:
        return 1 / (1 + np.exp(-input))

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        p = self.forward(input)
        return grad_output * (p * (1 - p))


class SoftMax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray, mode: str = 'train') -> np.ndarray:
        exp = np.exp(input - np.max(input))
        for i in range(len(input)):
            exp[i] /= np.sum(exp[i])
        return exp

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        p = self.forward(input)
        return grad_output * (p * (1 - p))
