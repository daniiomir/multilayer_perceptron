import gc
import numpy as np
from src.tools import save, load
from src.layers import Dropout


class Model:
    def __init__(self):
        self.network = []
        self.forward_list = []
        self.trainable = True

    def __len__(self):
        return len(self.network)

    @property
    def params(self):
        return [i for i in self.network if i.require_grad is True]

    def train_mode(self):
        self.trainable = True

    def test_mode(self):
        self.trainable = False

    def add_layer(self, layer):
        self.network.append(layer)

    def forward(self, input: np.ndarray, collect_garbage=False):
        self.forward_list.append(input)
        for layer in self.network:
            if isinstance(layer, Dropout) and not self.trainable:
                continue
            self.forward_list.append(layer.forward(self.forward_list[-1]))
        if self.trainable:
            assert len(self.forward_list) == len(self.network) + 1
        pred = self.forward_list[-1]
        if not self.trainable:
            self.clear_cache(collect_garbage)
        return pred

    def backward(self, layer_inputs, loss_grad):
        for layer_index in range(len(self.network))[::-1]:
            layer = self.network[layer_index]
            loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)

    def clear_cache(self, collect_garbage=False):
        self.forward_list.clear()
        if collect_garbage:
            for layer in self.params:
                layer.weights_grad = None
                layer.biases_grad = None
            gc.collect()

    def save_weights(self, path: str = 'tmp/weights.pkl'):
        weights_list = []
        for layer in self.params:
            weights_list.append((layer.weights, layer.biases))
        save(weights_list, path)

    def load_weights(self, path: str = 'tmp/weights.pkl'):
        weights_list = load(path)
        if isinstance(weights_list, list):
            assert len(weights_list) == len(self.params)
            for index, layer in enumerate(self.params):
                layer.weights = weights_list[index][0]
                layer.biases = weights_list[index][1]
        else:
            raise Exception('No weights in file!')
