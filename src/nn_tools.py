import os
import random
import numpy as np
from sklearn.model_selection import train_test_split


class EarlyStopping:
    def __init__(self, esr: int = 10):
        self.esr = esr
        self.current_esr = 0
        self.loss_list = []

    def add_loss(self, loss: float) -> None:
        self.loss_list.append(loss)

    def check_stop_training(self) -> bool:
        if self.current_esr > self.esr:
            print(f'Current early stopping rate bigger than {self.esr}. Stop training!')
            return True
        if len(self.loss_list) > 1:
            if min(self.loss_list[:-1]) < self.loss_list[-1]:
                self.current_esr += 1
            else:
                self.current_esr = 0
        return False


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def dataloader(inputs: np.ndarray, targets: np.ndarray, batchsize: int = 32,
               shuffle: bool = False) -> (np.ndarray, np.ndarray):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):  # tqdm.t
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def init_weights(params: list, weights_method: str = 'xavier_normal', biases_method: str = 'zeros') -> None:
    for layer in params:
        if weights_method == 'zeros':
            layer.weights = np.zeros(layer.weights_shape)
        elif weights_method == 'normal':
            layer.weights = np.random.normal(loc=0., scale=1., size=layer.weights_shape)
        elif weights_method == 'xavier_normal':
            layer.weights = np.random.normal(loc=0., scale=(1. / np.sqrt(layer.weights_shape[0])), size=layer.weights_shape)
        elif weights_method == 'kaiming_normal':
            layer.weights = np.random.normal(loc=0., scale=(np.sqrt(2. / layer.weights_shape[0])), size=layer.weights_shape)
        else:
            raise NotImplementedError

        if biases_method == 'zeros':
            layer.biases = np.zeros(layer.bias_shape)
        elif biases_method == 'ones':
            layer.biases = np.ones(layer.bias_shape)
        elif biases_method == 'normal':
            layer.biases = np.random.normal(loc=0., scale=1., size=layer.bias_shape)
        else:
            raise NotImplementedError


def clip_gradients(params: list, grad_clip: float = 3.):
    for layer in params:
        layer.weights_grad = np.clip(layer.weights_grad, -grad_clip, grad_clip)
        layer.biases_grad = np.clip(layer.biases_grad, -grad_clip, grad_clip)


def check_best_model(metric_list: list, current_metric: float) -> bool:
    if len(metric_list) == 0:
        return True
    if min(metric_list) > current_metric:
        return True
    return False


def threshold_prediction(array: np.ndarray, thrs: float = 0.5):
    return (array > thrs).astype(int)


def split_to_train_val_test(x, y, test_size, seed):
    X_train_full, X_test, y_train_full, y_test = train_test_split(x, y, test_size=float(test_size),
                                                                  random_state=int(seed))
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=float(test_size),
                                                      random_state=int(seed))
    return X_train, y_train, X_val, y_val, X_test, y_test
