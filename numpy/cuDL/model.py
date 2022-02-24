import re
import joblib
from cuDL.layer import Layer

from cuDL.loss import get_loss
from cuDL.optimizer import get_optimizer
from cuDL.activations import get_activation
from cuDL.metrics import get_metric

from sklearn.model_selection import train_test_split
import numpy as np
from numpy.random import shuffle as np_shuffle
from tqdm import tqdm


class Model:
    def __init__(self, name="", layers=None, loss=None, optimizer=None):

        self.name = name
        self.layers = layers
        if self.loss is not None:
            assert isinstance(
                self.loss, str
            ), "Loss name passed to model init must be a string"
            self.loss = get_loss(self.loss)

        if self.optimizer is not None:
            assert isinstance(
                self.optimizer, str
            ), "Optimizer name passed to model init must be a string"
            self.optimizer = get_optimizer(self.optimizer)

    def add_layer(self, layer):
        if self.layers is None:
            self.layers = []
        assert isinstance(layer, Layer), "Layer must be of type Layer"

        # lets just append layers and do the sanity check when compiling the model
        self.layers.append(layer)

    # TODO: change loss and optimizer to some useful defaults
    def compile(self, loss=None, optimizer=None):
        if loss is None and self.loss is None:
            raise ValueError(
                "No loss specified, You should specify a loss during model init or compile"
            )

        if optimizer is None and self.optimizer is None:
            raise ValueError(
                "No optimizer specified, You should specify an optimizer during model init or compile"
            )

        if self.layers is None or len(self.layers) == 0:
            raise ValueError(
                "No layers specified, You should specify at least one layer during model init with model add_layer before compile"
            )

        # sanity check the layers
        prev_layer = None
        for layer in self.layers:
            layer.compile(prev_layer)
            prev_layer = layer

        # if user specifies a loss/optimizer during compile, use it and override the ones during init
        if self.loss is not None:
            assert isinstance(
                self.loss, str
            ), "Loss name passed to model init must be a string"
            self.loss = get_loss(self.loss)

        if self.optimizer is not None:
            assert isinstance(
                self.optimizer, str
            ), "Optimizer name passed to model init must be a string"
            self.optimizer = get_optimizer(self.optimizer)

        return self

    def predict(self, X_test):
        return self.forward(X_test)

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        val_split=0.1,
        epochs=1,
        batch_size=1,
        seed=42,
        shuffle=True,
    ):

        self.mean_train_loss = []
        self.mean_val_loss = []
        self.mean_val_metric = []

        # sanity checks
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert (
            isinstance(val_split, float) and val_split >= 0 and val_split <= 1
        ), "val_split must be a float between 0 and 1"
        assert (
            isinstance(epochs, int) and epochs > 0
        ), "epochs must be a positive integer"
        assert (
            isinstance(batch_size, int) and batch_size > 0
        ), "batch_size must be a positive integer"
        assert isinstance(seed, int) and seed >= 0, "seed must be a positive integer"
        assert isinstance(shuffle, bool), "shuffle must be a boolean"

        if X_val is None and y_val is None:
            print(
                f"No validation data provided, splitting {val_split} of training data for validation"
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_split, random_state=seed
            )
        else:
            X_train, X_val, y_train, y_val = X, X_val, y, y_val

        # print data stats
        print(f"Training data shape: {X_train.shape}")
        print(f"Training data labels shape: {y_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Validation data labels shape: {y_val.shape}")

        if shuffle:
            X_train, y_train = np_shuffle(X_train, y_train)

        for epoch in tqdm(range(epochs)):
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]
                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_batch, y_pred)
                grad = self.backward(y_pred - y_batch)
                self.update(grad)

                self.mean_train_loss.append(loss)

            for i in range(0, X_val.shape[0], batch_size):
                X_batch = X_val[i : i + batch_size]
                y_batch = y_val[i : i + batch_size]
                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_batch, y_pred)
                self.mean_val_loss.append(loss)

    def forward(self, _input, *args, **kwargs):
        output = _input
        for layer in self.layers:
            output = layer.forward(output, *args, **kwargs)
        return output

    def backward(self, _input, *args, **kwargs):
        grad = _input
        for layer in reversed(self.layers):
            grad = layer.backward(grad, *args, **kwargs)
        return grad

    def compute_loss(self, y_true, y_pred):
        assert self.loss is not None, "No loss specified"
        return self.loss.forward(y_true, y_pred)

    def update(self, *args, **kwargs):
        assert self.optimizer is not None, "No optimizer specified"
        self.optimizer.update(self, *args, **kwargs)

    def summary(self):
        print("Model:", self.name)
        for layer in self.layers:
            print(layer.summary())
        print("Loss:", self.loss.summary())
        print("Optimizer:", self.optimizer.summary())

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(self, path):
        self = joblib.load(path)
        return self

    def __str__(self):
        return "Model: {}\nLayers: {}\nLoss: {}\nOptimizer: {}".format(
            self.name, self.layers, self.loss, self.optimizer
        )
