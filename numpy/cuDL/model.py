import re
from turtle import st
import joblib
from cuDL.layers import Layer

from cuDL.loss import get_loss
from cuDL.optim import get_optimizer
from cuDL.activations import get_activation
from cuDL.metrics import get_metric
from cuDL.regularizers import get_regularizer
import cuDL.metrics as metrics

from sklearn.model_selection import train_test_split
import numpy as np
from numpy.random import shuffle as np_shuffle
from tqdm import tqdm

from cuDL.utils import millify

import warnings

warnings.filterwarnings("ignore")


class Model:
    def __init__(
        self, name="", layers=None, loss=None, optimizer=None, model_save_path=None
    ):

        self.name = name
        self.layers = layers
        if loss is not None:
            assert isinstance(
                self.loss, str
            ), "Loss name passed to model init must be a string"
            self.loss = get_loss(self.loss)
        else:
            self.loss = None

        if optimizer is not None:
            assert isinstance(
                self.optimizer, str
            ), "Optimizer name passed to model init must be a string"
            self.optimizer = get_optimizer(self.optimizer)
        else:
            self.optimizer = None

        self.metrics = None
        self.regularizer = None
        self.model_save_path = model_save_path

        self.val_metric_to_track = None
        self.val_metric_to_track_mode = None

    def add_layer(self, layer):
        if self.layers is None:
            self.layers = []
        assert isinstance(layer, Layer), "Layer must be of type Layer"

        # lets just append layers and do the sanity check when compiling the model
        self.layers.append(layer)

    # TODO: change loss and optimizer to some useful defaults
    def compile(
        self,
        loss=None,
        optimizer=None,
        learning_rate=0.001,
        metrics=None,
        regularizer=None,
        weight_decay_rate=0.0,
        val_metric_to_track="loss",
        val_metric_to_track_mode="min",
        **kwargs,
    ):
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

        if regularizer is not None:
            assert isinstance(regularizer, str), "Regularizer name must be a string"
            # assert weight_decay_rate , "Weight decay rate must be greater than 0"
            self.regularizer = get_regularizer(regularizer, weight_decay_rate)

        self.loss = loss
        self.optimizer = optimizer

        self.val_metric_to_track = val_metric_to_track
        self.val_metric_to_track_mode = val_metric_to_track_mode

        if metrics is not None:
            self.metrics = []
            for metric in metrics:
                # appending name of the metric and the function
                self.metrics.append((metric, get_metric(metric)))

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
            # TODO: feels loose, we might need nestrov, momentum and adam based params.
            # perhaps we can set good defaults in optimizers directly but need to find a cleaner way
            self.optimizer = get_optimizer(
                self.optimizer, learning_rate=learning_rate, **kwargs
            )

        return self

    def predict(self, X_test, y_test=None, print_classification_metrics=False):
        y_pred = self.forward(X_test)
        num_classes = y_pred.shape[1]
        if y_test is not None:
            if print_classification_metrics:
                # show this for 2 decimal places
                print("Test Accuracy: {:.2f}".format(metrics.accuracy(y_test, y_pred)))
                y_test = y_test.argmax(axis=1)
                y_pred = y_pred.argmax(axis=1)
                print(metrics._classification_report(y_test, y_pred))
                print(metrics._confusion_matrix(y_test, y_pred))

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

        if self.val_metric_to_track_mode == "min":
            best_metric = np.inf
            compare_func = np.less
        elif self.val_metric_to_track_mode == "max":
            best_metric = -np.inf
            compare_func = np.greater

        if self.model_save_path is None:
            self.model_save_path = "./models/ckpt.bin"
            print(
                "No model save path specified, saving to default path: {}".format(
                    self.model_save_path
                )
            )

        def clear_lists():
            self.mean_train_loss = []
            self.mean_val_loss = []

            self.mean_val_metrics = {}

            if self.metrics is not None and self.metrics != []:
                for (metric_name, _) in self.metrics:
                    self.mean_val_metrics[metric_name] = []

        clear_lists()

        if self.val_metric_to_track == "loss":
            self.val_metric_to_track = "val_loss"
        elif self.val_metric_to_track not in [
            metric_name for (metric_name, _) in self.metrics
        ]:
            raise ValueError(
                f"Validation metric to track {self.val_metric_to_track} not in metrics {self.metrics}"
            )
        else:
            self.val_metric_to_track = "val_" + self.val_metric_to_track

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
            idxs = np.arange(X_train.shape[0])
            np_shuffle(idxs)
            X_train = X_train[idxs]
            y_train = y_train[idxs]

        tk = tqdm(range(epochs))
        for epoch in tk:
            for i in range(0, X_train.shape[0], batch_size):
                # # skip last batch if it is not full
                # if i + batch_size > X_train.shape[0]:
                #     break
                X_batch = X_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                y_pred = self.forward(X_batch)

                _loss = self.compute_loss(y_batch, y_pred)

                self.mean_train_loss.append(_loss)

                # backprop and update gradients
                params, grads = self.backward(y_batch, y_pred)
                self.update(params, grads)
                self.zero_grad()

            for i in range(0, X_val.shape[0], batch_size):
                # # skip last batch if it is not full
                # if i + batch_size > X_val.shape[0]:
                #     break
                X_batch = X_val[i : i + batch_size]
                y_batch = y_val[i : i + batch_size]
                y_pred = self.forward(X_batch)
                _loss = self.compute_loss(y_batch, y_pred)
                self.mean_val_loss.append(_loss)

                if self.metrics is not None and self.metrics != []:
                    for (metric_name, metric_func) in self.metrics:
                        self.mean_val_metrics[metric_name].append(
                            metric_func(y_batch, y_pred)
                        )

            train_loss = np.mean(self.mean_train_loss)
            val_loss = np.mean(self.mean_val_loss)

            printing_dict = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 2),
                "val_loss": round(val_loss, 2),
            }
            # print(self.metrics)
            if self.metrics is not None and self.metrics != []:
                for (metric_name, _) in self.metrics:
                    printing_dict["val_" + metric_name] = round(
                        np.mean(self.mean_val_metrics[metric_name]), 2
                    )
            # adds metrics to tqdm progress bar instead of printing
            tk.set_postfix(printing_dict)

            # save model if validation metric improved
            if compare_func(printing_dict[self.val_metric_to_track], best_metric):
                best_metric = printing_dict[self.val_metric_to_track]
                print(f"Saving model with {self.val_metric_to_track} {best_metric}.")
                if self.model_save_path is not None:
                    self.save(self.model_save_path)

            # clear the lists for the next epoch
            clear_lists()

    def forward(self, _input, *args, **kwargs):
        output = _input
        for layer in self.layers:
            output = layer.forward(output, *args, **kwargs)
        return output

    def backward(self, y_true, y_pred, *args, **kwargs):
        next_grad = self.loss.backward(y_true, y_pred)
        for layer in self.layers[::-1]:
            next_grad = layer.backward(next_grad, self.regularizer)

        params = []
        grads = []
        for layer in self.layers:
            params += layer.params
            grads += layer.grads

        return params, grads

    def compute_loss(self, y_true, y_pred):
        assert self.loss is not None, "No loss specified"
        loss = self.loss.forward(y_true, y_pred)
        if self.regularizer is not None:
            loss += self.regularizer(self.params)
        return loss

    def update(self, params, grads):
        assert self.optimizer is not None, "No optimizer specified"
        self.optimizer.update(params, grads)

    def summary(self, batch_size=-1):
        total_params = 0
        # print a summary of the model like keras model_summary
        print("Model Summary")
        print("=" * 30)
        print(f"Layers: {len(self.layers)}")
        for idx, layer in enumerate(self.layers):
            print(f"Layer {idx}: {layer.__class__.__name__}", end="\t")
            print(f"Input shape: ({batch_size, layer.input_dim})", end="\t")
            print(f"Output shape: ({batch_size, layer.output_dim})", end="\t")
            print(f"Activation: {layer.activation.name}")
            print(f"Params: {layer.num_params}")
            total_params += layer.num_params

        print(f"Total params: {millify(total_params)}")
        print("=" * 30)

    def save(self, path):
        # save the model parameters
        with open(path, "wb") as f:
            joblib.dump(self, f)

    @classmethod
    def load(self, path):
        # load the model parameters
        with open(path, "rb") as f:
            return joblib.load(f)

    @property
    def params(self):
        params = []
        for layer in self.layers:
            params += layer.params
        return params

    def __str__(self):
        return "Model: {}\nLayers: {}\nLoss: {}\nOptimizer: {}".format(
            self.name, self.layers, self.loss, self.optimizer
        )

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
