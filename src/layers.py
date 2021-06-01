import numpy as np
from typing import Union, Tuple


class Layer:
    def __init__(self):
        self.shape = None
        self.require_grad = False
        self.name = None
        self.weights = None
        self.biases = None
        self.weights_grad = None
        self.biases_grad = None

    def forward(self, input: np.ndarray, mode: str = 'train') -> np.ndarray:
        raise NotImplementedError

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        Performs a backpropagation step through the layer, with respect to the given input.
        To compute loss gradients w.r.t input, we need to apply chain rule (backprop):
        d loss / dx  = (d loss / d layer) * (d layer / dx)
        Luckily, we already receive d loss / d layer as input, so you only need to multiply it by d layer / d x.
        If our layer has parameters (e.g. dense layer), we also need to update them here using d loss / d layer
        """
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, input_units: int, output_units: int, name: str = 'Dense'):
        super().__init__()
        self.shape = (input_units, output_units)
        self.require_grad = True
        self.name = name

    def forward(self, input: np.ndarray, mode: str = 'train') -> np.ndarray:
        return np.dot(input, self.weights) + self.biases

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute df / dx = df / d dense * d dense / dx
        Where d dense / dx = weights transposed
        """
        grad_input = np.dot(grad_output, self.weights.T)

        self.weights_grad = np.dot(input.T, grad_output)
        self.biases_grad = grad_output.mean(axis=0)

        assert self.weights_grad.shape == self.weights.shape and self.biases_grad.shape == self.biases.shape
        return grad_input


class Conv2D(Layer):
    def __init__(self,
                 input_units: int,
                 output_units: int,
                 kernel: Tuple[int, int] = (3, 3),
                 padding: Union[int, Tuple[int, int]] = 0,
                 stride: Union[int, Tuple[int, int]] = 1,
                 name: str = 'Conv'):
        super().__init__()
        self.require_grad = True
        self.name = name

    def forward(self, input: np.ndarray, mode: str = 'train') -> np.ndarray:
        pass

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        pass


class MaxPooling(Layer):
    def __init__(self,
                 input_units: int,
                 output_units: int,
                 kernel: Tuple[int, int] = (3, 3)):
        super().__init__()

    def forward(self, input: np.ndarray, mode: str = 'train') -> np.ndarray:
        pass

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        pass


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray, mode: str = 'train') -> np.ndarray:
        self.shape = input.shape
        return np.ravel(input).reshape(input.shape[0], -1)

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        return grad_output.reshape(self.shape)


class Dropout(Layer):
    """
    "Dropout:  A Simple Way to Prevent Neural Networks from Overfitting"
    https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, input: np.ndarray, mode: str = 'train') -> np.ndarray:
        if mode == 'train':
            self.mask = np.random.binomial(1, self.p, input.shape) / self.p
            return input * self.mask
        else:
            return input

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.mask
