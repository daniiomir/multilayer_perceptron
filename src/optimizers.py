import numpy as np


class Optimizer:
    def __init__(self, params, learning_rate):
        self.model_params = params
        self.learning_rate = learning_rate

    def step(self, iter_num):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, learning_rate, weights_decay=0.3):
        super().__init__(params, learning_rate)
        self.model_params = params
        self.learning_rate = learning_rate
        self.weight_decay = weights_decay

    def step(self, iter_num):
        for layer in self.model_params:
            layer.weights += self.learning_rate * layer.weights_grad + self.weight_decay * layer.weights
            layer.biases += self.learning_rate * layer.biases_grad + self.weight_decay * layer.biases


class Momentum(Optimizer):
    def __init__(self, params, learning_rate, momentum=0.99):
        super().__init__(params, learning_rate)
        self.model_params = params
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weights_velocities = {i: np.zeros_like(self.model_params[i].weights) for i in
                                   range(len(self.model_params))}
        self.biases_velocities = {i: np.zeros_like(self.model_params[i].biases) for i in
                                  range(len(self.model_params))}

    def step(self, iter_num):
        for index, layer in enumerate(self.model_params):
            self.weights_velocities[index] = self.momentum * self.weights_velocities[index] + \
                                             self.learning_rate * layer.weights_grad
            self.biases_velocities[index] = self.momentum * self.biases_velocities[index] + \
                                            self.learning_rate * layer.biases_grad
            layer.weights -= self.weights_velocities[index]
            layer.biases -= self.biases_velocities[index]


class RMSProp(Optimizer):
    def __init__(self, params, learning_rate, eps=1e-8, alpha=0.9):
        super().__init__(params, learning_rate)
        self.model_params = params
        self.learning_rate = learning_rate
        self.eps = eps
        self.alpha = alpha
        self.weights_cache = {i: np.zeros_like(self.model_params[i].weights) for i in
                                   range(len(self.model_params))}
        self.biases_cache = {i: np.zeros_like(self.model_params[i].biases) for i in
                                  range(len(self.model_params))}

    def step(self, iter_num):
        for index, layer in enumerate(self.model_params):
            self.weights_cache[index] = self.alpha * self.weights_cache[index] + (1. - self.alpha) * (layer.weights_grad ** 2)
            self.biases_cache[index] = self.alpha * self.biases_cache[index] + (1. - self.alpha) * (layer.biases_grad ** 2)
            layer.weights -= self.learning_rate * (layer.weights_grad / (np.sqrt(self.weights_cache[index]) + self.eps))
            layer.biases -= self.learning_rate * (layer.biases_grad / (np.sqrt(self.biases_cache[index]) + self.eps))


class Adam(Optimizer):
    """
    "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"
    https://arxiv.org/pdf/1412.6980.pdf
    """
    def __init__(self, params, learning_rate, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, learning_rate)
        self.model_params = params
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps

        self.weights_m = {i: np.zeros_like(self.model_params[i].weights) for i in
                              range(len(self.model_params))}
        self.biases_m = {i: np.zeros_like(self.model_params[i].biases) for i in
                             range(len(self.model_params))}

        self.weights_v = {i: np.zeros_like(self.model_params[i].weights) for i in
                                   range(len(self.model_params))}
        self.biases_v = {i: np.zeros_like(self.model_params[i].biases) for i in
                                  range(len(self.model_params))}

    def step(self, iter_num):
        for index, layer in enumerate(self.model_params):
            self.weights_m[index] = self.betas[0] * self.weights_m[index] + (1. - self.betas[0]) * layer.weights_grad
            self.biases_m[index] = self.betas[0] * self.biases_m[index] + (1. - self.betas[0]) * layer.biases_grad

            self.weights_v[index] = self.betas[1] * self.weights_v[index] + (1. - self.betas[1]) * layer.weights_grad ** 2
            self.biases_v[index] = self.betas[1] * self.biases_v[index] + (1. - self.betas[1]) * layer.biases_grad ** 2

            weights_m_hat = self.weights_m[index] / (1. - self.betas[0] ** iter_num)
            biases_m_hat = self.biases_m[index] / (1. - self.betas[0] ** iter_num)

            weights_v_hat = self.weights_v[index] / (1. - self.betas[1] ** iter_num)
            biases_v_hat = self.biases_v[index] / (1. - self.betas[1] ** iter_num)

            layer.weights -= self.learning_rate * weights_m_hat / (np.sqrt(weights_v_hat) + self.eps)
            layer.biases -= self.learning_rate * biases_m_hat / (np.sqrt(biases_v_hat) + self.eps)
