import gc
import numpy as np
from src.tools import save, load


class Model:
    def __init__(self):
        self.network = []
        self.forward_list = []

    def __len__(self):
        return len(self.network)

    @property
    def params(self):
        return [i for i in self.network if i.require_grad is True]

    def add_layer(self, layer):
        self.network.append(layer)

    def forward(self, input: np.ndarray, mode='test', collect_garbage=False):
        self.forward_list.append(input)
        for layer in self.network:
            self.forward_list.append(layer.forward(self.forward_list[-1]))
        assert len(self.forward_list) == len(self.network) + 1
        pred = self.forward_list[-1]
        if mode == 'test':
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
