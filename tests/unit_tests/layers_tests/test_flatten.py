import numpy as np
from src.layers import Flatten


def test_flatten_forward1():
    input = np.zeros(shape=(3, 3, 28, 28))
    output = np.zeros(shape=(3, 3 * 28 * 28))
    l = Flatten()
    assert l.forward(input).shape == output.shape


def test_flatten_forward2():
    input = np.zeros(shape=(1, 1, 28, 28))
    output = np.zeros(shape=(1, 1 * 28 * 28))
    l = Flatten()
    assert l.forward(input).shape == output.shape


def test_flatten_backward1():
    input = np.zeros(shape=(3, 3, 28, 28))
    output = np.zeros(shape=(3, 3 * 28 * 28))
    l = Flatten()
    l.forward(input)
    result = l.backward(input=None, grad_output=output)
    assert result.shape == input.shape


def test_flatten_backward2():
    input = np.zeros(shape=(1, 1, 28, 28))
    output = np.zeros(shape=(1, 1 * 28 * 28))
    l = Flatten()
    l.forward(input)
    result = l.backward(input=None, grad_output=output)
    assert result.shape == input.shape
