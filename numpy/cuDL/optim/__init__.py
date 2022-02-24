import numpy as np
import abc

eps = 1e-8


def get_optimizer(name, learning_rate):
    name = name.lower()
    if name == "sgd":
        return SGD(lr=learning_rate)
    elif name == "adam":
        return Adam(lr=learning_rate)
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


class Adam(Optimizer):
    def __init__(
        self,
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        eps=eps,
        clip_norm=None,
        lr_scheduler=None,
        **kwargs,
    ):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.clip_norm = clip_norm
        self.lr_scheduler = lr_scheduler
        self.iter = 0
        self.m = None
        self.v = None
        self.t = None
        self.b1_cache = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v, self.t = [], [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
                self.t.append(np.zeros_like(param))
        self.iter += 1
        lr = self.lr
        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(self.iter)
        for param, grad, m, v, t in zip(params, grads, self.m, self.v, self.t):
            m = (self.beta_1 * m) + ((1 - self.beta_1) * grad)
            v = (self.beta_2 * v) + ((1 - self.beta_2) * (grad**2))
            t = t + 1
            param -= lr * m / (np.sqrt(v) + self.eps)
        if self.clip_norm is not None:
            norm = np.sqrt(sum([np.sum(param**2) for param in params]))
            if norm > self.clip_norm:
                for param in params:
                    param *= self.clip_norm / norm
        return params
