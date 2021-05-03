import numpy as np


class BinaryCrossEntropy:
    def __init__(self, eps=1e-5):
        self.eps = eps

    def __call__(self, y_true, y_pred):
        target = y_true
        output = y_pred

        output = np.clip(output, self.eps, 1.0 - self.eps)
        output = -target * np.log(output) - (1.0 - target) * np.log(1.0 - output)
        return np.mean(output, axis=-1)

    def backward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        return - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)


class MSE:
    def __call__(self, y_true, y_pred):
        return np.mean(np.square(y_pred - y_true), axis=-1)

    def backward(self, y_true, y_pred):
        return -(y_true - y_pred)
