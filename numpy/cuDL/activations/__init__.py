import numpy as np
import abc


def get_activation(name):
    name = name.lower()
    if name == "sigmoid":
        return Sigmoid()
    elif name == "linear":
        return Linear()
    elif name == "softmax":
        return Softmax()
    elif name == "relu":
        return Relu()
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
        self.output = x
        return x

    def backward(self):
        return np.ones_like(self.output)

    def __call__(self, x):
        return self.forward(x)


class Sigmoid(Activation):
    def __init__(self):
        self.name = "sigmoid"

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self):
        return self.output * (1 - self.output)

    def __call__(self, x):
        return self.forward(x)


class Softmax(Activation):
    def __init__(self):
        self.name = "softmax"

    def forward(self, x):
        self.output = x
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def backward(self):
        return np.ones(self.output.shape)

    def __call__(self, x):
        return self.forward(x)


class Relu(Activation):
    def __init__(self):
        self.name = "relu"

    def forward(self, x):
        self.output = x
        return np.maximum(x, 0)

    def backward(self):
        return np.where(self.output > 0, 1, 0)

    def __call__(self, x):
        return self.forward(x)
