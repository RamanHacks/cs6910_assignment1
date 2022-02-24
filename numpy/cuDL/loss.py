import numpy as np
import abc


def get_loss(name):
    name = name.lower()
    if name == "mse":
        return MSE()
    elif name == "cross_entropy":
        return CrossEntropy()
    else:
        raise ValueError(f"Unknown loss: {name}")


class Loss(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, y_true, y_pred):
        pass

    @abc.abstractmethod
    def backward(self, y_true, y_pred):
        pass

    @abc.abstractmethod
    def __call__(self, y_true, y_pred):
        pass


class MSE(Loss):
    def forward(self, y_true, y_pred):
        return 1 / 2 * np.mean(np.sum((y_true - y_pred) ** 2, axis=1))

    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]

    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)


class CrossEntropy(Loss):
    def __init__(self, eps=1e-8):
        self.eps = eps

    def forward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        return y_pred - y_true

    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)
