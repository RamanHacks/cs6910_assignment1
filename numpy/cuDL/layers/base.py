from abc import ABC, abstractmethod


class Layer(ABC):
    # to check if this is the first layer in the stack of X layers (where X can be dense/conv, etc)
    is_first_layer = False

    @abstractmethod
    def forward(self, input, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, pre_grad, *args, **kwargs):
        pass

    @abstractmethod
    # useful in model.compile to connect the layers and propogate the outputs during forward pass and gradients during backward pass
    def compile(self, prev_layer=None):
        pass

    @property
    @abstractmethod
    def grads(self):
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def reset_params(self):
        pass
