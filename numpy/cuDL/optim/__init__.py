import numpy as np
import abc


def get_optimizer(name, learning_rate):
    name = name.lower()
    if name == "sgd":
        return SGD(lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


class Optimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, params, grads):
        pass


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.lr * grad
