import cupy as cp

class Activation:
    def __call__(self, x):
        pass

    def backward(self, x):
        pass

class Sigmoid(Activation):
    # https://stackoverflow.com/a/23194336/12732481
    def __call__(self, x):
        """Numerically stable sigmoid function."""
        xp = cp.get_array_module(x)
        x = xp.clip(x, -500, 500)
        return 1 / (1 + xp.exp(-x))

    def backward(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class Linear(Activation):
    def __call__(self, x):
        xp = cp.get_array_module(x)
        return xp.asarray(x)

    def backward(self, x):
        return cp.ones_like(x)

class ReLU(Activation):
    def __call__(self, x):
        xp = cp.get_array_module(x)
        return xp.maximum(0, x)

    def backward(self, x):
        return cp.where(x > 0, cp.ones_like(x), cp.zeros_like(x))

class Tanh(Activation):
    def __call__(self, x):
        xp = cp.get_array_module(x)
        return xp.tanh(x)

    def backward(self, x):
        return 1 - self.__call__(x) ** 2

class Mish(Activation):
    def __call__(self, x):
        xp = cp.get_array_module(x)
        return x * xp.tanh(xp.log(1 + xp.exp(x)))
    
    def backward(self, x):
        return (cp.exp(x) * (4*cp.exp(2*x) + cp.exp(3*x) + 4*(1+x) + cp.exp(x)*(6+4*x))) / cp.power(2 + 2*cp.exp(x) + cp.exp(2*x), 2)

class SELU(Activation):
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946        
    
    def __call__(self, x):
        xp = cp.get_array_module(x)
        return self.scale * xp.where(x >= 0.0, x, self.alpha * (xp.exp(x) - 1.0))

    def backward(self, x):
        return self.scale * cp.where(x >= 0.0, 1.0, self.alpha * cp.exp(x))

class SiLU(Activation):
    def __call__(self, x):
        xp = cp.get_array_module(x)
        return x / (1 + xp.exp(-x))
    
    def backward(self, x):
        return (1 + cp.exp(-x) + x * cp.exp(-x)) / cp.power(1 + cp.exp(-x), 2)

        