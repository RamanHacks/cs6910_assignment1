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
        **kwargs,
    ):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        # self.iter = 0
        self.m = None
        self.v = None
        self.t = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v, self.t = [], [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
                self.t.append(np.zeros_like(param))
        # self.iter += 1
        lr = self.lr

        for param, grad, m, v, t in zip(params, grads, self.m, self.v, self.t):
            m = (self.beta_1 * m) + ((1 - self.beta_1) * grad)
            v = (self.beta_2 * v) + ((1 - self.beta_2) * (grad**2))

            t = t + 1

            # add bias correction
            # results are slightly better without bias correction
            # TODO: need to double check
            m_hat = m / (1 - self.beta_1**t)
            v_hat = v / (1 - self.beta_2**t)

            param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params
