import numpy as np
import abc

from scipy.special import logsumexp


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
    elif name == "tanh":
        return Tanh()
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
    def __init__(self):
        self.name = "linear"

    def forward(self, x):
        self.output = x
        return x

    def backward(self):
        return np.ones_like(self.output).astype(np.float32)

    def __call__(self, x):
        return self.forward(x)


class Sigmoid(Activation):
    def __init__(self):
        self.name = "sigmoid"

    def forward(self, x):
        x = 1 / (1 + np.exp(-x))
        self.output = x
        return x

    def backward(self):
        return self.output * (1 - self.output)

    def __call__(self, x):
        return self.forward(x)


# class Softmax(Activation):
#     def __init__(self):
#         self.name = "softmax"

#     def forward(self, x):
#         # https://cs231n.github.io/linear-classify/#softmax
#         # using a normalizing trick to avoid overflow
#         # https://stackoverflow.com/questions/42599498/numercially-stable-softmax/42606665#42606665
#         x -= np.max(x, axis=1, keepdims=True)
#         x = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
#         self.output = x
#         return x

#     # def backward(self):

#     #     # https://www.bragitoff.com/2021/12/efficient-implementation-of-softmax-activation-function-and-its-derivative-jacobian-in-python/
#     #     """Returns the jacobian of the Softmax function for the given set of inputs.
#     #     Inputs:
#     #     x: should be a 2d array where the rows correspond to the samples
#     #         and the columns correspond to the nodes.
#     #     Returns: jacobian
#     #     """
#     #     s = self.output
#     #     a = np.eye(s.shape[-1])
#     #     temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=np.float32)
#     #     temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=np.float32)
#     #     temp1 = np.einsum("ij,jk->ijk", s, a)
#     #     temp2 = np.einsum("ij,ik->ijk", s, s)
#     #     return temp1 - temp2

#     def backward(self, grad):
#         # https://sgugger.github.io/a-simple-neural-net-in-numpy.html#a-simple-neural-net-in-numpy
#         return self.output * (grad - (grad * self.output).sum(axis=1)[:, None])

#     def __call__(self, x):
#         return self.forward(x)


class Softmax(Activation):
    def __init__(self):
        self.name = "softmax"

    def forward(self, x):
        # https://github.com/scipy/scipy/blob/b5d8bab88af61d61de09641243848df63380a67f/scipy/special/_logsumexp.py#L130
        # computing with log-sum-exp trick for numerical stability
        # x = np.exp(x - logsumexp(x, axis=1, keepdims=True))
        # self.output = np.copy(x)
        x -= np.max(x, axis=1, keepdims=True)
        x = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
        self.output = x

        return x

    def backward(self, grad):
        # https://sgugger.github.io/a-simple-neural-net-in-numpy.html#a-simple-neural-net-in-numpy
        return self.output * (grad - (grad * self.output).sum(axis=1)[:, None])

    def __call__(self, x):
        return self.forward(x)


class Tanh(Activation):
    def __init__(self):
        self.name = "tanh"

    def forward(self, x):
        x = np.tanh(x)
        self.output = x
        return x

    def backward(self):
        return 1 - self.output ** 2

    def __call__(self, x):
        return self.forward(x)


class Relu(Activation):
    def __init__(self):
        self.name = "relu"

    def forward(self, x):
        self.output = np.copy(x)
        return np.maximum(x, 0)

    def backward(self):
        return np.where(self.output > 0, 1, 0).astype(np.float32)

    def __call__(self, x):
        return self.forward(x)
