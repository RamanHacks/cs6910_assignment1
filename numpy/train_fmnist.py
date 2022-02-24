# test fashion mnist with cuDL

from ast import Store
from genericpath import exists
import os
import sys
import time
import argparse
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from cuDL.layers import Dense
from cuDL.model import Model


def download_fmnist(data_dir):
    """
    Download the fashion mnist dataset
    """
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Reshape and normalize the data
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Convert the labels to categorical one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def plot_images(images, cls_true, cls_pred=None, smooth=True):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on the Fashion MNIST dataset"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory to store the models",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="fashion_mnist_model",
        help="Name of the model",
    )
    # training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split for training data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="Shuffle the training data",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of layers in the model",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=str,
        default="128",
        help="this is an integer or a comma-separated list of integers indicating the number of hidden units in each layer of the model.",
    )
    parser.add_argument(
        "--weight_decay_rate",
        type=float,
        default=0.0005,
        help="Weight decay rate (L2 regularizer) for training",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="Optimizer to use for training",
    )
    parser.add_argument(
        "--weight_init",
        type=str,
        default="xavier",
        help="Weight initialization scheme",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function to use",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="cross_entropy",
        help="Loss function to use",
    )
    parser.add_argument(
        "--plot",
        type=bool,
        default=False,
        help="Plot the training data",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    str_hidden_sizes = args.hidden_sizes
    args.hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    # if only one number is passed, that means user wants to use the same number of hidden units in each layer
    if len(args.hidden_sizes) == 1:
        args.hidden_sizes = [args.hidden_sizes[0]] * args.num_layers
    assert len(args.hidden_sizes) == args.num_layers

    if args.model_dir is None and not args.debug:
        # set a descriptive model directory name
        model_args = [
            "fmnist",
            "L_{}".format(args.num_layers),
            "HS_{}".format(str_hidden_sizes),
            "DR_{}".format(args.weight_decay_rate),
            "O_{}".format(args.optimizer),
            "loss_{}".format(args.loss),
        ]
        model_str = "_".join(model_args)
        args.model_dir = os.path.join(
            f"models/{model_str}",
        )
        os.makedirs(args.model_dir, exist_ok=True)

        save_path = os.path.join(args.model_dir, "ckpt.bin")

        print("Saving models here: {}".format(save_path))
    # Set the random seed for reproducibility
    np.random.seed(args.seed)

    # print summary of the arguments
    print("=" * 30)
    print("Arguments:")
    # debug: False
    ignore_args = ["model_dir", "model_name", "debug", "plot"]
    for arg in vars(args):
        if arg not in ignore_args:
            print("{}: {}".format(arg, getattr(args, arg)))
    print("=" * 30)

    # Download the Fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = download_fmnist(args.model_dir)

    if args.debug:
        x_train = x_train[:100]
        y_train = y_train[:100]

    # Plot some examples of the training data
    if args.plot:
        plot_images(x_train, y_train)

    # Build the model
    model = Model(model_save_path=save_path)

    model.add_layer(
        Dense(
            input_dim=x_train.shape[1],
            output_dim=args.hidden_sizes[0],
            activation=args.activation,
            init_method=args.weight_init,
        )
    )

    for idx in range(args.num_layers - 1):  # exclude the first layer
        model.add_layer(
            Dense(
                input_dim=args.hidden_sizes[idx],
                output_dim=args.hidden_sizes[idx + 1],
                activation=args.activation,
                init_method=args.weight_init,
            )
        )

    if args.loss == "cross_entropy":
        model.add_layer(
            Dense(
                input_dim=args.hidden_sizes[-1],
                output_dim=10,
                activation="softmax",
                init_method=args.weight_init,
            )
        )
    elif args.loss == "mse":
        model.add_layer(
            Dense(
                input_dim=args.hidden_size,
                output_dim=10,
                activation="linear",
                init_method=args.weight_init,
            )
        )

    model.summary(args.batch_size)

    loss = args.loss
    optimizer = args.optimizer

    # compile the model
    model.compile(
        loss=loss,
        optimizer=optimizer,
        learning_rate=args.learning_rate,
        metrics=["accuracy"],
        regularizer="L2",
        weight_decay_rate=args.weight_decay_rate,
    )

    # Train the model
    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_split=args.val_split,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    print("=" * 30)
    print("Evaluating model on test data with best model")

    # load the best model
    model.load(model.model_save_path)

    # Evaluate the model
    model.predict(x_test, y_test, print_classification_metrics=True)