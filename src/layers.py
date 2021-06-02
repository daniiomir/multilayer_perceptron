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
                 kernel: int = 3,
                 padding: int = 0,
                 stride: int = 1,
                 name: str = 'Conv'):
        super().__init__()
        self.require_grad = True
        self.name = name

    def forward(self, input: np.ndarray, mode: str = 'train') -> np.ndarray:
        pass

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        pass


class MaxPooling2D(Layer):
    """
    Pooling layer.

    Do downsamling of input matrix by finding max elements in kernel region.

    forward pass:
        input: matrix with same width and height - shape (batch_size, channels, width, height)
        output: formula for output matrix width and height = ((width - kernel + 2 * padding) / stride) + 1
    """
    def __init__(self,
                 kernel: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 padding_mode: str = 'constant'):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode

    def forward(self, input: np.ndarray, mode: str = 'train') -> np.ndarray:
        assert input.shape[2] >= self.kernel and input.shape[3] >= self.kernel

        if self.padding:
            input = np.pad(input, (self.padding,), self.padding_mode)

        output_shape = int((input.shape[2] - self.kernel + 2 * self.padding) / self.stride) + 1
        output = np.zeros((input.shape[0], input.shape[1], output_shape, output_shape))

        for i in range(output.shape[2]):
            for j in range(output.shape[3]):
                start_h = i * self.stride
                end_h = start_h + self.kernel
                start_w = j * self.stride
                end_w = start_w + self.kernel
                window = input[:, :, start_h:end_h, start_w:end_w]
                output[:, :, i, j] = np.max(window, axis=(2, 3))
        return output

    def backward(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        grad_input = np.zeros_like(input)
        for i in range(grad_output.shape[2]):
            for j in range(grad_output.shape[3]):
                start_h = i * self.stride
                end_h = start_h + self.kernel
                start_w = j * self.stride
                end_w = start_w + self.kernel
                window = input[:, :, start_h:end_h, start_w:end_w]
                mask = (window == np.max(window, axis=(2, 3)))
                grad_input[:, :, start_h:end_h, start_w:end_w] = mask * grad_output[:, :, i, j]
        return grad_input


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
