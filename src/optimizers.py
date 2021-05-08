import numpy as np


class Optimizer:
    def __init__(self, params, learning_rate):
        self.model_params = params
        self.learning_rate = learning_rate

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, learning_rate, grad_clip=5):
        super().__init__(params, learning_rate)
        self.model_params = params
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip

    def step(self):
        for layer in self.model_params:
            layer.weights -= self.learning_rate * np.clip(layer.weights_grad, -self.grad_clip, self.grad_clip)
            layer.biases -= self.learning_rate * np.clip(layer.biases_grad, -self.grad_clip, self.grad_clip)


class Momentum(Optimizer):
    def __init__(self, params, learning_rate, momentum=0.99, grad_clip=5):
        super().__init__(params, learning_rate)
        self.model_params = params
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.grad_clip = grad_clip
        self.weights_velocities = {i: np.zeros_like(self.model_params[i].weights) for i in
                                   range(len(self.model_params))}
        self.biases_velocities = {i: np.zeros_like(self.model_params[i].biases) for i in
                                  range(len(self.model_params))}

    def step(self):
        for index, layer in enumerate(self.model_params):
            self.weights_velocities[index] = self.momentum * self.weights_velocities[index] + \
                                             self.learning_rate * np.clip(layer.weights_grad, -self.grad_clip, self.grad_clip)
            self.biases_velocities[index] = self.momentum * self.biases_velocities[index] + \
                                            self.learning_rate * np.clip(layer.biases_grad, -self.grad_clip, self.grad_clip)
            layer.weights -= self.weights_velocities[index]
            layer.biases -= self.biases_velocities[index]


# TODO : добавить оптимизаторы RMSprop, ADAgrad, Adam
