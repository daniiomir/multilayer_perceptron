import numpy as np
from numpy.lib.stride_tricks import as_strided
from src.layers import MaxPooling2D

# pic1_4x4_channel1 + prev_grad + after_grad

pic1_4x4_channel1 = np.array([
    [
        [[0], [1], [2], [3]],
        [[4], [5], [6], [7]],
        [[8], [9], [10], [11]],
        [[12], [13], [14], [15]]
    ]
], dtype=np.float64)

# pool2x2

pool2x2_grad_pic1_4x4_channel1 = np.ones(shape=(1, 1, 3, 3))
pool2x2_backward_result_pic1_4x4_channel1 = np.array([
    [
        [[0], [0], [0], [0]],
        [[0], [1], [1], [1]],
        [[0], [1], [1], [1]],
        [[0], [1], [1], [1]]
    ]
], dtype=np.float64)
pool2x2_backward_result_pic1_4x4_channel1 = np.moveaxis(pool2x2_backward_result_pic1_4x4_channel1, 3, 1)

# pool3x3

pool3x3_grad_pic1_4x4_channel1 = np.ones(shape=(1, 1, 2, 2))
pool3x3_backward_result_pic1_4x4_channel1 = np.array([
    [
        [[0], [0], [0], [0]],
        [[0], [0], [0], [0]],
        [[0], [0], [1], [1]],
        [[0], [0], [1], [1]]
    ]
], dtype=np.float64)
pool3x3_backward_result_pic1_4x4_channel1 = np.moveaxis(pool3x3_backward_result_pic1_4x4_channel1, 3, 1)

# pool4x4

pool4x4_grad_pic1_4x4_channel1 = np.ones(shape=(1, 1, 1, 1))
pool4x4_backward_result_pic1_4x4_channel1 = np.array([
    [
        [[0], [0], [0], [0]],
        [[0], [0], [0], [0]],
        [[0], [0], [0], [0]],
        [[0], [0], [0], [1]]
    ]
], dtype=np.float64)
pool4x4_backward_result_pic1_4x4_channel1 = np.moveaxis(pool4x4_backward_result_pic1_4x4_channel1, 3, 1)

###########

pic3_4x4_channel1 = np.array([
    [
        [[0], [1], [2], [3]],
        [[4], [5], [6], [7]],
        [[8], [9], [10], [11]],
        [[12], [13], [14], [15]]
    ],
    [
        [[0], [1], [2], [3]],
        [[4], [5], [6], [7]],
        [[8], [9], [10], [11]],
        [[12], [13], [14], [15]]
    ],
    [
        [[0], [1], [2], [3]],
        [[4], [5], [6], [7]],
        [[8], [9], [10], [11]],
        [[12], [13], [14], [15]]
    ]
])

pic1_4x4_channel3 = np.array([
    [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]],
        [[25, 26, 27], [28, 29, 30], [31, 32, 33], [34, 34, 36]],
        [[37, 38, 39], [40, 41, 42], [43, 44, 45], [46, 47, 48]],
    ]
])

pic3_4x4_channel3 = np.array([
    [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]],
        [[25, 26, 27], [28, 29, 30], [31, 32, 33], [34, 34, 36]],
        [[37, 38, 39], [40, 41, 42], [43, 44, 45], [46, 47, 48]],
    ],
    [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]],
        [[25, 26, 27], [28, 29, 30], [31, 32, 33], [34, 34, 36]],
        [[37, 38, 39], [40, 41, 42], [43, 44, 45], [46, 47, 48]],
    ],
    [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]],
        [[25, 26, 27], [28, 29, 30], [31, 32, 33], [34, 34, 36]],
        [[37, 38, 39], [40, 41, 42], [43, 44, 45], [46, 47, 48]],
    ]
])

# shape of imgs - (batch_size, height, width, channels)

pic1_4x4_channel1 = np.moveaxis(pic1_4x4_channel1, 3, 1)  # shape - (1, 1, 4, 4)
pic1_4x4_channel3 = np.moveaxis(pic1_4x4_channel3, 3, 1)  # shape - (1, 3, 4, 4)
pic3_4x4_channel1 = np.moveaxis(pic3_4x4_channel1, 3, 1)  # shape - (3, 1, 3, 3)
pic3_4x4_channel3 = np.moveaxis(pic3_4x4_channel3, 3, 1)  # shape - (3, 3, 3, 3)


# now shape of imgs - (batch_size, channels, height, width)


def pool2d(A, kernel_size, stride, padding):
    """
    POOLING STACK OVERFLOW VERSION USING NUMPY.LIB.STRIDE_TRICKS.AS_STRIDED

    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    if padding:
        A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[2] - kernel_size) // stride + 1,
                    (A.shape[3] - kernel_size) // stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A,
                     shape=(A.shape[0], A.shape[1], *output_shape, *kernel_size),
                     strides=(A.strides[0], A.strides[1],
                              stride * A.strides[2],
                              stride * A.strides[3],
                              A.strides[2], A.strides[3])
                     )
    return A_w.max(axis=(4, 5))


# forward tests
# pic1_channel1


def test_pool2x2_forward_pic1_channel1():
    pool = MaxPooling2D(kernel=2, stride=1, padding=0)
    result = pool.forward(pic1_4x4_channel1)
    real = pool2d(A=pic1_4x4_channel1, kernel_size=2, stride=1, padding=0)
    assert (result == real).all()


def test_pool3x3_forward_pic1_channel1():
    pool = MaxPooling2D(kernel=3, stride=1, padding=0)
    result = pool.forward(pic1_4x4_channel1)
    real = pool2d(A=pic1_4x4_channel1, kernel_size=3, stride=1, padding=0)
    assert (result == real).all()


def test_pool4x4_forward_pic1_channel1():
    pool = MaxPooling2D(kernel=4, stride=1, padding=0)
    result = pool.forward(pic1_4x4_channel1)
    real = pool2d(A=pic1_4x4_channel1, kernel_size=4, stride=1, padding=0)
    assert (result == real).all()


# pic3_channel1


def test_pool2x2_forward_pic3_channel1():
    pool = MaxPooling2D(kernel=2, stride=1, padding=0)
    result = pool.forward(pic3_4x4_channel1)
    real = pool2d(A=pic3_4x4_channel1, kernel_size=2, stride=1, padding=0)
    assert (result == real).all()


def test_pool3x3_forward_pic3_channel1():
    pool = MaxPooling2D(kernel=3, stride=1, padding=0)
    result = pool.forward(pic3_4x4_channel1)
    real = pool2d(A=pic3_4x4_channel1, kernel_size=3, stride=1, padding=0)
    assert (result == real).all()


def test_pool4x4_forward_pic3_channel1():
    pool = MaxPooling2D(kernel=4, stride=1, padding=0)
    result = pool.forward(pic3_4x4_channel1)
    real = pool2d(A=pic3_4x4_channel1, kernel_size=4, stride=1, padding=0)
    assert (result == real).all()


# pic1_channel3


def test_pool2x2_forward_pic1_channel3():
    pool = MaxPooling2D(kernel=2, stride=1, padding=0)
    result = pool.forward(pic1_4x4_channel3)
    real = pool2d(A=pic1_4x4_channel3, kernel_size=2, stride=1, padding=0)
    assert (result == real).all()


def test_pool3x3_forward_pic1_channel3():
    pool = MaxPooling2D(kernel=3, stride=1, padding=0)
    result = pool.forward(pic1_4x4_channel3)
    real = pool2d(A=pic1_4x4_channel3, kernel_size=3, stride=1, padding=0)
    assert (result == real).all()


def test_pool4x4_forward_pic1_channel3():
    pool = MaxPooling2D(kernel=4, stride=1, padding=0)
    result = pool.forward(pic1_4x4_channel3)
    real = pool2d(A=pic1_4x4_channel3, kernel_size=4, stride=1, padding=0)
    assert (result == real).all()


# pic3_channel3


def test_pool2x2_forward_pic3_channel3():
    pool = MaxPooling2D(kernel=2, stride=1, padding=0)
    result = pool.forward(pic3_4x4_channel3)
    real = pool2d(A=pic3_4x4_channel3, kernel_size=2, stride=1, padding=0)
    assert (result == real).all()


def test_pool3x3_forward_pic3_channel3():
    pool = MaxPooling2D(kernel=3, stride=1, padding=0)
    result = pool.forward(pic3_4x4_channel3)
    real = pool2d(A=pic3_4x4_channel3, kernel_size=3, stride=1, padding=0)
    assert (result == real).all()


def test_pool4x4_forward_pic3_channel3():
    pool = MaxPooling2D(kernel=4, stride=1, padding=0)
    result = pool.forward(pic3_4x4_channel3)
    real = pool2d(A=pic3_4x4_channel3, kernel_size=4, stride=1, padding=0)
    assert (result == real).all()


# random arrays
# pic1_256x256_channel3


def test_pool2x2_forward_pic1_256x256_channel3():
    pic = np.random.randint(low=0, high=255, size=(1, 3, 256, 256))
    pool = MaxPooling2D(kernel=2, stride=1, padding=0)
    result = pool.forward(pic)
    real = pool2d(A=pic, kernel_size=2, stride=1, padding=0)
    assert (result == real).all()


def test_pool3x3_forward_pic1_256x256_channel3():
    pic = np.random.randint(low=0, high=255, size=(1, 3, 256, 256))
    pool = MaxPooling2D(kernel=3, stride=1, padding=0)
    result = pool.forward(pic)
    real = pool2d(A=pic, kernel_size=3, stride=1, padding=0)
    assert (result == real).all()


def test_pool4x4_forward_pic1_256x256_channel3():
    pic = np.random.randint(low=0, high=255, size=(1, 3, 256, 256))
    pool = MaxPooling2D(kernel=4, stride=1, padding=0)
    result = pool.forward(pic)
    real = pool2d(A=pic, kernel_size=4, stride=1, padding=0)
    assert (result == real).all()


# pic3_256x256_channel3


def test_pool2x2_forward_pic3_256x256_channel3():
    pic = np.random.randint(low=0, high=255, size=(3, 3, 256, 256))
    pool = MaxPooling2D(kernel=2, stride=1, padding=0)
    result = pool.forward(pic)
    real = pool2d(A=pic, kernel_size=2, stride=1, padding=0)
    assert (result == real).all()


def test_pool3x3_forward_pic3_256x256_channel3():
    pic = np.random.randint(low=0, high=255, size=(3, 3, 256, 256))
    pool = MaxPooling2D(kernel=3, stride=1, padding=0)
    result = pool.forward(pic)
    real = pool2d(A=pic, kernel_size=3, stride=1, padding=0)
    assert (result == real).all()


def test_pool4x4_forward_pic3_256x256_channel3():
    pic = np.random.randint(low=0, high=255, size=(3, 3, 256, 256))
    pool = MaxPooling2D(kernel=4, stride=1, padding=0)
    result = pool.forward(pic)
    real = pool2d(A=pic, kernel_size=4, stride=1, padding=0)
    assert (result == real).all()


# pic1_512x512_channel3


def test_pool2x2_forward_pic1_512x512_channel3():
    pic = np.random.randint(low=0, high=255, size=(1, 3, 512, 512))
    pool = MaxPooling2D(kernel=2, stride=1, padding=0)
    result = pool.forward(pic)
    real = pool2d(A=pic, kernel_size=2, stride=1, padding=0)
    assert (result == real).all()


def test_pool3x3_forward_pic1_512x512_channel3():
    pic = np.random.randint(low=0, high=255, size=(1, 3, 512, 512))
    pool = MaxPooling2D(kernel=3, stride=1, padding=0)
    result = pool.forward(pic)
    real = pool2d(A=pic, kernel_size=3, stride=1, padding=0)
    assert (result == real).all()


def test_pool4x4_forward_pic1_512x512_channel3():
    pic = np.random.randint(low=0, high=255, size=(1, 3, 512, 512))
    pool = MaxPooling2D(kernel=4, stride=1, padding=0)
    result = pool.forward(pic)
    real = pool2d(A=pic, kernel_size=4, stride=1, padding=0)
    assert (result == real).all()


# backward tests
# pic1_4x4_channel1

def test_pool2x2_backward_pic1_4x4_channel1():
    pool = MaxPooling2D(kernel=2, stride=1, padding=0)
    result = pool.backward(pic1_4x4_channel1, pool2x2_grad_pic1_4x4_channel1)
    assert (result == pool2x2_backward_result_pic1_4x4_channel1).all()


def test_pool3x3_backward_pic1_4x4_channel1():
    pool = MaxPooling2D(kernel=3, stride=1, padding=0)
    result = pool.backward(pic1_4x4_channel1, pool3x3_grad_pic1_4x4_channel1)
    assert (result == pool3x3_backward_result_pic1_4x4_channel1).all()


def test_pool4x4_backward_pic1_4x4_channel1():
    pool = MaxPooling2D(kernel=4, stride=1, padding=0)
    result = pool.backward(pic1_4x4_channel1, pool4x4_grad_pic1_4x4_channel1)
    assert (result == pool4x4_backward_result_pic1_4x4_channel1).all()