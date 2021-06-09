import numpy as np
from src.modules.model import Model


class Loss:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        raise NotImplementedError

    def backward(self, model: Model, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        raise NotImplementedError


class BinaryCrossEntropyLoss(Loss):
    """
    Loss for binary classification

    y_pred: [0.5, ..., 0.1] | shape - (probabilities,)
    y_true: [0.5, ..., 0.1] | shape - (probabilities,)
    """
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        target = y_true
        output = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return -1 * np.mean(target * np.log(output) + (1 - target) * np.log(1 - output))

    def backward(self, model: Model, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        loss_grad = y_pred - y_true[:, np.newaxis]
        model.backward(model.forward_list, loss_grad)


class CrossEntropyLoss(Loss):
    """
    Loss for multiclass classification

    y_true: [[0, 0, 1], [...], [...]] | one hot encoded labels | shape - (batch_size, num_classes)
    y_pred: [[0.1, 0.3, 0.6], [...], [...]] | shape - (batch_size, num_classes)
    """
    def __init__(self):
        super().__init__()

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        output = np.clip(y_pred, self.eps, 1. - self.eps)
        output /= output.sum(axis=1)[:, np.newaxis]
        loss = -(y_true * np.log(output)).sum(axis=1)
        return np.mean(loss)

    def backward(self, model: Model, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        loss_grad = y_pred - y_true
        model.backward(model.forward_list, loss_grad)


class MeanSquaredErrorLoss(Loss):
    def __call__(self, y_true, y_pred):
        return np.mean(np.square(y_pred - y_true[:, np.newaxis]))

    def backward(self, model: Model, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        loss_grad = -2 * (y_true[:, np.newaxis] - y_pred)
        model.backward(model.forward_list, loss_grad)
