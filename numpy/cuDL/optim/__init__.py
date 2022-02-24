from asyncio import new_event_loop
import enum
from hashlib import new
import numpy as np
import abc

eps = 1e-8


def get_optimizer(name, learning_rate, **kwargs):
    name = name.lower()
    if name == "sgd":
        return SGD(lr=learning_rate, **kwargs)
    elif name == "rmsprop":
        return RMSprop(lr=learning_rate, **kwargs)
    elif name == "adam" or name == "nadam":
        return Adam(lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


class Optimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, params, grads):
        pass


class SGD(Optimizer):
    def __init__(self, lr=0.01, weight_decay_rate=0.0, momentum=0.0, nesterov=False):
        self.name = "sgd"
        self.lr = lr
        self._lambda = weight_decay_rate

        self.momentum = momentum
        self.nesterov = nesterov

        self.history = None

    def update(self, params, grads):

        if self.history is None:
            self.history = [np.zeros_like(p) for p in params]
        # weight decay
        if self._lambda != 0:
            for param, grad in zip(params, grads):
                m = param.shape[-1]
                grad += self._lambda * param / m
        # for nag, instead of computing gradient at look ahead weights, we use bengio's trick to compute gradient at current weights
        # explanation: https://ruder.io/optimizing-gradient-descent/#nadam (Ctrl + F "Dozat")
        if self.momentum != 0 and self.nesterov:
            for idx, (param, grad, history) in enumerate(
                zip(params, grads, self.history)
            ):
                history = self.momentum * history + self.lr * grad
                self.history[idx] = history
                param -= self.momentum * history + self.lr * grad

        elif self.momentum != 0:
            for idx, (param, grad, history) in enumerate(
                zip(params, grads, self.history)
            ):
                history = self.momentum * history + self.lr * grad
                self.history[idx] = history
                param -= history
        else:
            # vanilla sgd
            for param, grad in zip(params, grads):
                param -= self.lr * grad


class RMSprop(Optimizer):
    def __init__(
        self, lr=0.01, momentum=0.9, eps=1e-8, weight_decay_rate=0.0, nesterov=False,
    ):
        self.name = "rmsprop"
        self.lr = lr
        self.momentum = momentum
        self.eps = eps
        self._lambda = weight_decay_rate

        self.history = None

    def update(self, params, grads):
        if self.history is None:
            self.history = [np.zeros_like(p) for p in params]
        # weight decay
        if self._lambda != 0:
            for param, grad in zip(params, grads):
                m = param.shape[-1]
                grad += self._lambda * param / m
        # update history
        for idx, (param, grad, history) in enumerate(zip(params, grads, self.history)):
            history = self.momentum * history + (1 - self.momentum) * grad ** 2
            param -= self.lr * grad / np.sqrt(history + self.eps)
            self.history[idx] = history


class Adam(Optimizer):
    def __init__(
        self,
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        eps=eps,
        weight_decay_rate=0.0,
        momentum=0.0,  # not used as beta1 is same as momentum. Will resolve later
        nesterov=False,
        **kwargs,
    ):
        self.name = "adam"
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        self._lambda = weight_decay_rate

        self.iter = 0
        self.m = None
        self.v = None
        self.nesterov = nesterov

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        self.iter += 1
        # bias_correction_constant = (
        #     self.lr
        #     * np.sqrt(1 - self.beta_2 ** self.iter)
        #     / (1 - self.beta_1 ** self.iter)
        # )

        # weight decay
        if self._lambda != 0:
            for param, grad in zip(params, grads):
                m = param.shape[-1]
                grad += self._lambda * param / m
        for idx, (param, grad, m, v) in enumerate(zip(params, grads, self.m, self.v)):
            m = (self.beta_1 * m) + ((1 - self.beta_1) * grad)
            v = (self.beta_2 * v) + ((1 - self.beta_2) * (grad ** 2))

            mhat = m / (1 - self.beta_1 ** self.iter)
            vhat = v / (1 - self.beta_2 ** self.iter)

            if self.nesterov:  # nadam
                # https://paperswithcode.com/method/nadam
                # https://ruder.io/optimizing-gradient-descent/index.html#nadam
                param -= (
                    self.lr
                    / (np.sqrt(vhat) + self.eps)
                    * (
                        self.beta_1 * mhat
                        + (1 - self.beta_1) * grad / (1 - self.beta_1 ** self.iter)
                    )
                )
            else:
                param -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

            # update m and v
            self.m[idx] = m
            self.v[idx] = v

        return params
