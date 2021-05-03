import tqdm
import numpy as np


class Model:
    def __init__(self):
        self.network = []
        self.forward_list = []

    def add_layer(self, layer):
        self.network.append(layer)

    def forward(self, X):
        input = X
        for l in self.network:
            self.forward_list.append(l.forward(input))
            input = self.forward_list[-1]
        assert len(self.forward_list) == len(self.network)
        return self.forward_list[-1]

    def backward(self, layer_inputs, loss_grad):
        for layer_index in range(len(self.network))[::-1]:
            layer = self.network[layer_index]
            loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)

    def loader(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.random.permutation(len(inputs))
        for start_idx in tqdm.trange(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    @property
    def params(self):
        return [i for i in self.network if i.require_grad is True]
