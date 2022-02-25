import cupy as cp
from .activation import ReLU
class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def apply(self, lr=None):
        if lr is not None:
            self.lr = lr
        pass


class SGD(Optimizer):
    def __init__(self, params, lr):
        super(SGD, self).__init__(params, lr)

    def apply(self, lr=None):
        if lr is not None:
            self.lr = lr
        for param in self.params.values():
            param["W"] -= self.lr * param["W_d"]
            param["b"] -= self.lr * param["b_d"]

class Momentum(Optimizer):
    def __init__(self, params, lr, mu):
        super(Momentum, self).__init__(params, lr)
        self.mu = mu
        for layer in list(self.params.values()):
            layer.update({"W_g": cp.zeros_like(layer["W_d"])})
            layer.update({"b_g": cp.zeros_like(layer["b_d"])})

    def apply(self, lr=None):
        if lr is not None:
            self.lr = lr
        for param in self.params.values():
            param["W_g"] = self.lr * param["W_d"] + self.mu * param["W_g"]
            param["W"] -=  param["W_g"]
            param["b_g"] = self.lr * param["b_d"] + self.mu * param["b_g"]
            param["b"] -=  param["b_g"]


class RMSProp(Optimizer):
    def __init__(self, params, lr=0.01, beta=0.99, eps=1e-7):
        super(RMSProp, self).__init__(params, lr)
        self.beta = beta
        self.eps = eps
        for layer in list(self.params.values()):
            layer.update({"W_s": cp.zeros_like(layer["W_d"])})
            layer.update({"b_s": cp.zeros_like(layer["b_d"])})

    def apply(self, lr=None):
        if lr is not None:
            self.lr = lr
        for param in self.params.values():
            param["W_s"] = self.beta * param["W_s"] + (1 - self.beta) * cp.square(param["W_d"])
            param["W"] -= self.lr * param["W_d"] / (cp.sqrt(param["W_s"]) + self.eps)
            param["b_s"] = self.beta * param["b_s"] + (1 - self.beta) * cp.square(param["b_d"])
            param["b"] -= self.lr * param["b_d"] / (cp.sqrt(param["b_s"]) + self.eps)


class AdaGrad(Optimizer):
    def __init__(self, params, lr=0.01, eps=1e-7):
        super(AdaGrad, self).__init__(params, lr)
        self.eps = eps
        for layer in list(self.params.values()):
            layer.update({"W_s": cp.zeros_like(layer["W_d"])})
            layer.update({"b_s": cp.zeros_like(layer["b_d"])})

    def apply(self, lr=None):
        if lr is not None:
            self.lr = lr
        for param in self.params.values():
            param["W_s"] = cp.square(param["W_d"])
            param["W"] -= self.lr * param["W_d"] / (cp.sqrt(param["W_s"]) + self.eps)
            param["b_s"] = cp.square(param["b_d"])
            param["b"] -= self.lr * param["b_d"] / (cp.sqrt(param["b_s"]) + self.eps)

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-7):
        super(Adam, self).__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.k = 1
        for layer in list(self.params.values()):
            layer.update({"W_m": cp.zeros_like(layer["W_d"])})
            layer.update({"W_v": cp.zeros_like(layer["W_d"])})
            layer.update({"b_m": cp.zeros_like(layer["b_d"])})
            layer.update({"b_v": cp.zeros_like(layer["b_d"])})

    def apply(self, lr=None):
        if lr is not None:
            self.lr = lr
        for param in self.params.values():
            param["W_m"] = (1 - self.beta1) * param["W_d"] + self.beta1 * param["W_m"]
            param["W_v"] = (1 - self.beta2) * cp.square(param["W_d"]) + self.beta2 * param["W_v"]
            w_m_hat = param["W_m"] / (1 - self.beta1 ** self.k)
            w_v_hat = param["W_v"] / (1 - self.beta2 ** self.k)
            param["W"] -= self.lr * w_m_hat / (cp.sqrt(w_v_hat) + self.eps)

            param["b_m"] = (1 - self.beta1) * param["b_d"] + self.beta1 * param["b_m"]
            param["b_v"] = (1 - self.beta2) * cp.square(param["b_d"]) + self.beta2 * param["b_v"]
            b_m_hat = param["b_m"] / (1 - self.beta1 ** self.k)
            b_v_hat = param["b_v"] / (1 - self.beta2 ** self.k)
            param["b"] -= self.lr * b_m_hat / (cp.sqrt(b_v_hat) + self.eps)
        self.k += 1


class Nadam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-7):
        super(Nadam, self).__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.k = 1
        for layer in list(self.params.values()):
            layer.update({"W_m": cp.zeros_like(layer["W_d"])})
            layer.update({"W_v": cp.zeros_like(layer["W_d"])})
            layer.update({"b_m": cp.zeros_like(layer["b_d"])})
            layer.update({"b_v": cp.zeros_like(layer["b_d"])})

    def apply(self, lr=None):
        if lr is not None:
            self.lr = lr
        for param in self.params.values():
            param["W_m"] = (1 - self.beta1) * param["W_d"] + self.beta1 * param["W_m"]
            param["W_v"] = (1 - self.beta2) * cp.square(param["W_d"]) + self.beta2 * param["W_v"]
            w_m_hat = param["W_m"] / (1 - self.beta1 ** self.k)
            w_v_hat = param["W_v"] / (1 - self.beta2 ** self.k)
            param["W"] -= self.lr / (cp.sqrt(w_v_hat) + self.eps) * (self.beta1 * w_m_hat + (1 - self.beta1)
                                                                           * param["W_d"] / (1 - self.beta1**self.k))

            param["b_m"] = (1 - self.beta1) * param["b_d"] + self.beta1 * param["b_m"]
            param["b_v"] = (1 - self.beta2) * cp.square(param["b_d"]) + self.beta2 * param["b_v"]
            b_m_hat = param["b_m"] / (1 - self.beta1 ** self.k)
            b_v_hat = param["b_v"] / (1 - self.beta2 ** self.k)
            param["b"] -= self.lr / (cp.sqrt(b_v_hat) + self.eps) * (self.beta1 * b_m_hat + (1 - self.beta1)
                                                                            * param["b_d"] / (1 - self.beta1**self.k))
        self.k += 1


class AdamW(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-7, weight_decay=0.05):
        super(AdamW, self).__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.k = 1
        for layer in list(self.params.values()):
            layer.update({"W_m": cp.zeros_like(layer["W_d"])})
            layer.update({"W_v": cp.zeros_like(layer["W_d"])})
            layer.update({"b_m": cp.zeros_like(layer["b_d"])})
            layer.update({"b_v": cp.zeros_like(layer["b_d"])})

    def apply(self, lr=None):
        if lr is not None:
            self.lr = lr
        for param in self.params.values():
            param["W_m"] = (1 - self.beta1) * param["W_d"] + self.beta1 * param["W_m"]
            param["W_v"] = (1 - self.beta2) * cp.square(param["W_d"]) + self.beta2 * param["W_v"]
            w_m_hat = param["W_m"] / (1 - self.beta1 ** self.k)
            w_v_hat = param["W_v"] / (1 - self.beta2 ** self.k)
            param["W"] -= self.lr * w_m_hat / (cp.sqrt(w_v_hat) + self.eps) + self.weight_decay * param["W_d"]

            param["b_m"] = (1 - self.beta1) * param["b_d"] + self.beta1 * param["b_m"]
            param["b_v"] = (1 - self.beta2) * cp.square(param["b_d"]) + self.beta2 * param["b_v"]
            b_m_hat = param["b_m"] / (1 - self.beta1 ** self.k)
            b_v_hat = param["b_v"] / (1 - self.beta2 ** self.k)
            param["b"] -= self.lr * b_m_hat / (cp.sqrt(b_v_hat) + self.eps)
        self.k += 1