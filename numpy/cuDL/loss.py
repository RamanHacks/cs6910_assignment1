from tkinter.tix import Y_REGION
import numpy as np
import abc

from cuDL.activations import get_activation


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
        return (y_pred - y_true) / y_true.shape[0]

    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)


class CrossEntropy(Loss):
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.softmax = get_activation("softmax")

    # def forward(self, y_true, y_pred):
    #     y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
    #     return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    # def backward(self, y_true, y_pred):
    #     y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
    #     # return y_pred - y_true
    #     return np.where(y_true == 1, -1 / y_pred, 0)

    def forward(self, y_true, y_pred):
        # in this version we will try softmax and then cross entropy
        # this makes it easier to differentiate

        # converting one-hot to labels
        y_true = y_true.argmax(axis=1)

        m = y_true.shape[0]

        y_pred = self.softmax(y_pred)

        log_likelihood = -np.log(y_pred[range(m), y_true])

        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, y_true, y_pred):
        # https://deepnotes.io/softmax-crossentropy
        y_true = y_true.argmax(axis=1)

        m = y_true.shape[0]
        grad = self.softmax(y_pred)
        grad[range(m), y_true] -= 1
        grad = grad / m
        return grad

    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)
