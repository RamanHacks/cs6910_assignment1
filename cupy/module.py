import cupy as cp
from .activation import Linear, ReLU, Sigmoid, Tanh, Mish, SELU, SiLU
from .initializer import Constant, Uniform, XavierNormal, Random, HeNormal
from copy import deepcopy as dc
from abc import ABC

class Layer:
    def __call__(self, **kwargs):
        raise NotImplementedError

    def backward(self, **x):
        raise NotImplementedError

class ParamLayer(Layer, ABC):
    def __init__(self,
                 weight_shape,
                 weight_initializer,
                 bias_initializer,
                 regularizer_type: str = None,
                 lamda: float = 0.
                 ):
        self.vars = {"W": weight_initializer.initialize(weight_shape),
                     "b": bias_initializer.initialize((1, weight_shape[1])),
                     "W_d": cp.zeros(weight_shape),
                     "b_d": cp.zeros((1, weight_shape[1]))}

        self.z = None
        self.input = None

        self.regularizer_type = regularizer_type  # noqa
        self.lamda = lamda

    def summary(self):
        name = self.__class__.__name__
        n_param = self.vars["W"].shape[0] * self.vars["W"].shape[1] + self.vars["b"].shape[1]
        output_shape = (None, self.vars["b"].shape[1])
        return name, output_shape, n_param

    @property
    def input_shape(self):
        return self.vars["W"].shape[0]


class Dense(ParamLayer, ABC):
    def __init__(self, in_feat: int,
                 out_feat: int,
                 activation = Linear(),
                 weight_initializer = Uniform(),
                 bias_initializer = Constant(),
                 regularizer_type: str = None,
                 lamda: float = 0.
                 ):
        super().__init__(weight_shape=(in_feat, out_feat),
                         weight_initializer=weight_initializer,
                         bias_initializer=bias_initializer,
                         regularizer_type=regularizer_type,
                         lamda=lamda
                         )
        
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.act = activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_type = regularizer_type
        self.lamda = lamda

    def __call__(self, x):
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        self.input = x
        z = cp.dot(x, self.vars["W"]) + self.vars["b"]
        self.z = z
        a = self.act(z)
        return a

    def backward(self, **grad):
        #  https://cs182sp21.github.io/static/slides/lec-5.pdf
        grad = grad["grad"]
        dz = grad * self.act.backward(self.z)
        self.vars["W_d"] = cp.dot(self.input.T, dz) / dz.shape[0]
        self.vars["b_d"] = cp.sum(dz, axis=0, keepdims=True) / dz.shape[0]

        if self.regularizer_type == "l2":
            self.vars["W_d"] += self.lamda * self.vars["W"]
            # Biases are not regularized: https://cs231n.github.io/neural-networks-2/#reg
            # self.vars["b_d"] += self.lamda * self.vars["b"]
        elif self.regularizer_type == "l1":
            self.vars["W_d"] += self.lamda
            # self.vars["b_d"] += self.lamda

        grad = dz.dot(self.vars["W"].T)
        
        return dict(grad=grad)

class Sequential:
    def __init__(self, *args):
        self._layers = args[0]
        self._parameters = {i: self._layers[i].vars for i in range(len(self._layers)) if
                            isinstance(self._layers[i], ParamLayer)}

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        for i,layer in enumerate(self._layers):
            self._parameters[i].update({'input':x})
            x = layer(x)
        return x

    @property
    def parameters(self):
        return self._parameters

    def backward(self, loss):
        grad = dict(grad=loss['grad'])
        for layer in self._layers[::-1]:
            grad = layer.backward(**grad)

    def set_weights(self, params):
        self._parameters = dc(params)
        for i, layer in enumerate(self._layers):
            if isinstance(self._layers[i], ParamLayer):
                self._layers[i].vars = self._parameters[i]       


class SimpleLinear(Sequential):
    def __init__(self, input_dim, out_dim, hidden_dim, hidden_layers,
                 activation, initializer, regularizer, lamda):
        layers = []
        act_map = {'relu': ReLU(), 'linear': Linear(), 'sigmoid': Sigmoid(), 'tanh': Tanh(), 'mish': Mish(), 'selu':SELU(), 'silu':SiLU()}
        init_map = {'uniform': Uniform(), 'xavier': XavierNormal(), 'he': HeNormal(ReLU()), 'random': Random(), 'zeros': Constant(0.)}
        activation = act_map[activation]
        initializer = init_map[initializer]
        layers.append(Dense(in_feat=input_dim,
                                    out_feat=hidden_dim[0],
                                    activation=activation,
                                    weight_initializer=initializer,
                                    bias_initializer=Constant(0.),
                                    regularizer_type=regularizer,
                                    lamda=lamda))
        for i in range(1, hidden_layers-1):
            layers.append(Dense(
                in_feat=hidden_dim[i-1],
                out_feat=hidden_dim[i],
                activation=activation,
                weight_initializer=initializer,
                bias_initializer=Constant(0.),
                regularizer_type=regularizer,
                lamda=lamda
            ))
            
        layers.append(Dense(in_feat=hidden_dim[-1],
                                 out_feat=out_dim,
                                 activation=Linear(),
                                 ))
        super().__init__(layers)