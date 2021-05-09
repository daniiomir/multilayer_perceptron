import os
import random
import numpy as np


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


def init_weights(params: list, method: str = 'xavier_normal') -> None:
    for layer in params:
        if method == 'zeros':
            layer.weights = np.zeros(layer.shape)
        elif method == 'normal':
            layer.weights = np.random.normal(loc=0., scale=1., size=layer.shape)
        elif method == 'xavier_normal':
            layer.weights = np.random.normal(loc=0., scale=(np.sqrt(layer.shape[0])), size=layer.shape)
        elif method == 'kaiming_normal':
            layer.weights = np.random.normal(loc=0., scale=(np.sqrt(2 / layer.shape[0])), size=layer.shape)
        else:
            raise NotImplementedError


def clip_gradients(params: list, grad_clip: int = 3):
    for layer in params:
        layer.weights_grad = np.clip(layer.weights_grad, -grad_clip, grad_clip)
        layer.biases_grad = np.clip(layer.biases_grad, -grad_clip, grad_clip)
