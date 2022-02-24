import numpy as np
import abc


def get_activation(name):
    name = name.lower()
    if name == "sigmoid":
        return Sigmoid()
    elif name == "linear":
        return Linear()
    else:
        raise ValueError(f"Unknown activation: {name}")


class Activation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def backward(self, x):
        pass

    @abc.abstractmethod
    def __call__(self, x):
        pass


class Linear(Activation):
    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)

    def __call__(self, x):
        return x


class Sigmoid(Activation):
    def __init__(self):
        self.name = "sigmoid"

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))

    def __call__(self, x):
        return self.forward(x)
