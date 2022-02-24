from turtle import forward
import numpy as np
import abc

# implement weight regularization


def get_regularizer(name, weight_decay=0.01):
    name = name.lower()
    if name == "l1":
        return L1(weight_decay)
    elif name == "l2":
        return L2(weight_decay)
    else:
        raise ValueError(f"Unknown regularizer: {name}")


class Regularizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, W):
        pass


class L1(Regularizer):
    def __init__(self, l=1e-4):
        self.l = l

    def forward(self, W):
        self.last_W = W
        return self.l * np.sum(np.abs(W))

    def backward(self, W=None):
        self.last_W = W if W is not None else self.last_W
        return self.l * np.sign(self.last_W)

    def __call__(self, W):
        return self.forward(W)


class L2(Regularizer):
    def __init__(self, l=0.01):
        self.l = l

    def forward(self, W):
        self.last_W = W
        return 0.5 * self.l * np.sum(W**2)

    def backward(self, W=None):
        self.last_W = W if W is not None else self.last_W
        return self.l * self.last_W

    def __call__(self, W):
        if isinstance(W, list):
            # this is a list of weights
            rloss = 0
            for w in W:
                rloss += self.forward(w)
            return rloss
        else:
            return self.forward(W)
