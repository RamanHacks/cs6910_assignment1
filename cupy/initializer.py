import cupy as cp
from .activation import Activation, ReLU

class Initializer:
    def initialize(self, x):
        pass


class Constant(Initializer):
    def __init__(self, constant=0):
        self._const = constant

    def initialize(self, shape):
        return self._const * cp.ones(shape)

class Uniform(Initializer):
    def initialize(self, shape):
        return cp.random.uniform(0, 1, shape)

class Random(Initializer):
    def initialize(self, shape):
        in_dim, out_dim = shape
        # Initialize with scaled (large) values:
        # https://datascience-enthusiast.com/DL/Improving-DeepNeural-Networks-Initialization.html
        return cp.random.randn(in_dim, out_dim) * 10 

class XavierNormal(Initializer):
    def initialize(self, shape):
        in_dim, out_dim = shape
        mu, sigma = 0, 0.1
        return cp.random.normal(mu, sigma, (in_dim, out_dim)) \
        * cp.sqrt(2.0 / (in_dim + out_dim))

class HeNormal(Initializer):
    def __init__(self, non_linearity, mode="fan_in"):
        if not isinstance(non_linearity, Activation):
            raise Exception()
        self.non_linearity = non_linearity
        self.mode = mode

    def initialize(self, shape):
        fan_in, fan_out = shape
        fan = fan_in if self.mode == "fan_in" else fan_out
        if isinstance(self.non_linearity, ReLU):
            gain = cp.sqrt(2)
        else:
            raise NotImplementedError
        std = gain / cp.sqrt(fan)
        return cp.random.normal(0, std, shape)