from .base import Layer
import numpy as np
from cuDL.activations import get_activation
from cuDL.initializers import get_initializer


class Dense(Layer):
    def __init__(
        self, input_dim, output_dim, activation="linear", init_method="xavier", name=""
    ):
        # sanity check
        assert isinstance(input_dim, int), "input_dim must be of type int"
        assert isinstance(output_dim, int), "output_dim must be of type int"
        assert (
            isinstance(activation, str) or activation is None
        ), "activation must be of type str or None"

        # generally its assumed that any layer is not the first layer and this is handled by the compile method
        self.is_first_layer = False
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = get_activation(activation)
        self.init_method = init_method
        self.name = name

        self.reset_params()

    def reset_params(self, init_method=None):
        if init_method is None and self.init_method is None:
            raise ValueError(
                "No init method specified, You should specify an init method during model init or when calling reset_params"
            )

        if init_method is not None:
            self.init_method = init_method

        self.weights = get_initializer(self.init_method)(
            self.input_dim, self.output_dim
        )
        # https://cs231n.github.io/neural-networks-2/ says its common to set bias to zero
        self.bias = get_initializer("zeros")(self.output_dim)

    def compile(self, prev_layer=None):
        if prev_layer is None:
            self.is_first_layer = True
        else:
            assert (
                prev_layer.output_dim == self.input_dim
            ), "input_dim of previous layer must be equal to output_dim of current layer"

    def forward(self, _input, *args, **kwargs):
        # sanity check
        assert isinstance(
            _input, np.ndarray
        ), "input to forward must be of type np.ndarray"
        assert (
            _input.shape[1] == self.input_dim
        ), f"input to forward must have {self.input_dim} columns"
        assert _input.shape[0] > 0, f"input to forward must have at least one row"

        self._input = np.copy(_input)

        outputs = np.dot(_input, self.weights) + self.bias
        outputs = self.activation(outputs)

        return outputs

    def backward(self, pre_grad, regularizer=None, *args, **kwargs):
        # sanity check
        assert isinstance(pre_grad, np.ndarray), "pre_grad must be of type np.ndarray"
        assert (
            pre_grad.shape[1] == self.output_dim
        ), f"pre_grad must have {self.output_dim} columns"
        assert pre_grad.shape[0] > 0, f"pre_grad must have at least one row"

        m = pre_grad.shape[1]

        # compute gradients
        # d_activation = self.activation.backward()

        # check if pred_grad and d_activation have dimesions
        # if pre_grad.ndim != d_activation.ndim:
        #     # for softmax activation
        #     d_activation = np.einsum("ijk,ik->ij", d_activation, pre_grad)
        if self.activation.name == "softmax":
            d_activation = self.activation.backward(pre_grad)
        else:
            d_activation = self.activation.backward()
            d_activation = pre_grad * d_activation

        self.d_weights = self._input.T.dot(d_activation)
        self.d_bias = d_activation.mean(axis=0)

        # print(d_activation.shape)
        # print(self._input.shape)

        # self.d_weights = 1 / m * (d_activation @ self._input.T)
        # self.d_bias = np.mean(d_activation, axis=1, keepdims=True)
        # self.d_bias = np.sum(d_activation, axis=1, keepdims=1)

        # # regularization gradients
        # if regularizer is not None:
        #     self.d_weights += regularizer.backward(self.weights)
        # self.d_bias += regularizer.backward(self.bias)

        # propagate gradients to lower layers
        if not self.is_first_layer:
            d_input = np.dot(d_activation, self.weights.T)
            # d_input = self.weights.T @ d_activation

            return d_input

    @property
    def params(self):
        return self.weights, self.bias

    @property
    # return sum of parameters of the model
    def num_params(self):
        return self.weights.size + self.bias.size

    @property
    def grads(self):
        return self.d_weights, self.bias

    def zero_grad(self):
        self.d_weights = np.ones_like(self.weights)
        self.d_bias = np.ones_like(self.bias)
