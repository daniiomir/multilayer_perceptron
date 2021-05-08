import numpy as np
from src.model import Model


class Loss:
    def __init__(self, eps: float = 1e-15):
        self.eps = eps

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        raise NotImplementedError

    def backward(self, model: Model, y_true: np.ndarray, y_pred: np.ndarray):
        raise NotImplementedError


class BinaryCrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        target = y_true
        output = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return -1 * np.mean(target * np.log(output) + (1 - target) * np.log(1 - output))

    def backward(self, model: Model, y_true: np.ndarray, y_pred: np.ndarray):
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        loss_grad = -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)
        model.backward(model.forward_list, loss_grad)


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        target = y_true[:, np.newaxis]
        if target.shape[1] == 1:
            target = np.append(1 - target, target, axis=1)
        output = np.clip(y_pred, self.eps, 1. - self.eps)
        output /= output.sum(axis=1)[:, np.newaxis]
        loss = -(target * np.log(output)).sum(axis=1)
        return np.mean(loss)

    def backward(self, model: Model, y_true: np.ndarray, y_pred: np.ndarray):
        loss_grad = y_pred - y_true[:, np.newaxis]
        model.backward(model.forward_list, loss_grad)


class MeanSquaredErrorLoss(Loss):
    def __call__(self, y_true, y_pred):
        return np.mean(np.square(y_pred - y_true), axis=-1)

    def backward(self, model: Model, y_true: np.ndarray, y_pred: np.ndarray):
        loss_grad = -(y_true - y_pred)
        model.backward(model.forward_list, loss_grad)
