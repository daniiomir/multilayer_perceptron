import numpy as np
from src.modules.layers import Conv2D

pic1_4x4_channel1 = np.array([
    [
        [[0], [1], [2], [3]],
        [[4], [5], [6], [7]],
        [[8], [9], [10], [11]],
        [[12], [13], [14], [15]]
    ]
], dtype=np.float64)

pool2x2_grad_pic1_4x4_channel1 = ...
pool2x2_backward_result_pic1_4x4_channel1 = ...
pool2x2_backward_result_pic1_4x4_channel1 = np.moveaxis(pool2x2_backward_result_pic1_4x4_channel1, 3, 1)


def test_conv2d_2x2_forward():
    l = Conv2D(1, 3, kernel=2, stride=1, padding=0)
    l.weights = np.ones(l.weights_shape)
    l.biases = np.ones(l.bias_shape)
    ...
