import enum
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
        weight_decay_rate=0.0,
        **kwargs,
    ):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        self._lambda = weight_decay_rate

        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        self.iter += 1
        bias_correction_constant = (
            self.lr * np.sqrt(1 - self.beta_2 ** self.iter) / (1 - self.beta_1 ** self.iter)
        )

        # weight decay
        if self._lambda != 0:
            for param, grad in zip(params, grads):
                grad += self._lambda * param
        for idx, (param, grad, m, v) in enumerate(zip(params, grads, self.m, self.v)):
            m = (self.beta_1 * m) + ((1 - self.beta_1) * grad)
            v = (self.beta_2 * v) + ((1 - self.beta_2) * (grad ** 2))

            # # add bias correction
            # # results are slightly better without bias correction
            # # TODO: need to double check
            # m_hat = m / (1 - self.beta_1 ** self.iter)
            # v_hat = v / (1 - self.beta_2 ** self.iter)

            param -= m * bias_correction_constant / (np.sqrt(v) + self.eps)

            # update m and v
            self.m[idx] = m
            self.v[idx] = v

        return params
