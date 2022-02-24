import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import os
import sys
import argparse
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

from tqdm import tqdm


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
    # y_train = to_categorical(y_train, 10)
    # y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a model on the Fashion MNIST dataset"
    )
    parser.add_argument(
        "--model_dir", type=str, default=None, help="Directory to store the models",
    )
    # training arguments
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split for training data",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for training",
    )
    parser.add_argument(
        "--shuffle", type=bool, default=True, help="Shuffle the training data",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for training",
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of layers in the model",
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
        "--optimizer", type=str, default="sgd", help="Optimizer to use for training",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.0, help="Momentum for SGD",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        default=False,
        help="Whether to use nesterov",
    )
    parser.add_argument(
        "--weight_init",
        type=str,
        default="xavier",
        help="Weight initialization scheme",
    )
    parser.add_argument(
        "--activation", type=str, default="relu", help="Activation function to use",
    )
    parser.add_argument(
        "--loss", type=str, default="cross_entropy", help="Loss function to use",
    )
    parser.add_argument(
        "--plot", type=bool, default=False, help="Plot the training data",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.momentum > 0.0 and args.optimizer not in ["sgd", "rmsprop"]:
        print("Momentum is only supported for SGD/RMSprop optimizer")
        sys.exit(1)

    if args.nesterov and args.optimizer not in ["sgd", "adam"]:
        print("nesterov is only supported for SGD/adam optimizer")
        sys.exit(1)

    if args.nesterov and args.momentum == 0.0:
        print("nesterov requires momentum")
        sys.exit(1)

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
        args.model_dir = os.path.join(f"models/{model_str}",)
        os.makedirs(args.model_dir, exist_ok=True)

        save_path = os.path.join(args.model_dir, "torch_ckpt.bin")

        print("Saving models here: {}".format(save_path))

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set the device to use for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download the Fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = download_fmnist(data_dir="data")

    # make crossentropy loss work with one hot encoding
    def cross_entropy_one_hot(_input, target):
        _, labels = target.max(dim=1)
        return nn.CrossEntropyLoss()(_input, labels)

    if args.loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "mse":
        criterion = torch.nn.MSELoss()
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        # final_activation = None

    # get activation from torch corresponding to the args activation
    if args.activation == "relu":
        activation = nn.ReLU()
    elif args.activation == "sigmoid":
        activation = nn.Sigmoid()
    elif args.activation == "tanh":
        activation = nn.Tanh()
    elif args.activation == "leaky_relu":
        activation = nn.LeakyReLU()
    elif args.activation == "elu":
        activation = nn.ELU()
    elif args.activation == "softplus":
        activation = nn.Softplus()
    elif args.activation == "softmax":
        activation = nn.Softmax()

    modules = []
    # create the layers
    modules.append(nn.Linear(x_train.shape[1], args.hidden_sizes[0]))
    modules.append(activation)
    for i in range(1, args.num_layers):
        modules.append(nn.Linear(args.hidden_sizes[i - 1], args.hidden_sizes[i]))
        modules.append(activation)

    modules.append(nn.Linear(args.hidden_sizes[-1], 10))
    # if final_activation is not None:
    #     modules.append(final_activation)

    # Create the model
    model = torch.nn.Sequential(*modules)

    model.to(device)
    print(model)

    # print number of parameters in the model
    print(
        "Number of parameters in the model:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # initialization function, first checks the module type,
    # then applies the desired changes to the weights
    def init_xavier_normal(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    # Initialize the weights
    if args.weight_init == "xavier":
        model.apply(init_xavier_normal)

    # Create the optimizer
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay_rate,
            momentum=args.momentum,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay_rate,
        )
    elif args.optimizer == "nadam":
        optimizer = torch.optim.NAdam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay_rate,
        )
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay_rate,
            alpha=args.momentum,
        )
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay_rate,
        )
    elif args.optimizer == "adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay_rate,
        )
    else:
        raise ValueError("Unknown optimizer")

    # get args.val_split of the training data for validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=args.val_split, random_state=args.seed
    )

    # convert to tensors
    x_train = torch.tensor(x_train).float()
    x_val = torch.tensor(x_val).float()
    x_test = torch.tensor(x_test).float()

    if args.loss == "cross_entropy":
        y_train = torch.tensor(y_train).long()
        y_val = torch.tensor(y_val).long()

        y_test = torch.tensor(y_test).long()
    else:
        y_train = torch.tensor(y_train).float()
        y_val = torch.tensor(y_val).float()
        y_test = torch.tensor(y_test).float()

    # print the shapes of the data
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_val shape:", x_val.shape)
    print("y_val shape:", y_val.shape)

    # Create the dataloaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=args.shuffle,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    train_loss = []
    val_loss = []
    val_accuracy = []

    best_val_accuracy = 0.0

    def train(args, model, device, train_loader, optimizer, epoch):
        train_loss = []
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        return np.mean(train_loss)

    def validate(args, model, device, val_loader, criterion):
        val_loss = []
        val_accuracy = []
        model.eval()

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)

                val_loss.append(loss.item())
                if args.loss == "cross_entropy":
                    val_accuracy.append(
                        np.mean(
                            np.argmax(y_pred.cpu().numpy(), axis=1)
                            == y_batch.cpu().numpy()
                        )
                    )
                else:
                    # print("Preds:", y_pred.cpu().numpy())
                    # print("Labels:", y_batch.cpu().numpy())
                    # print("Preds:", np.argmax(y_pred.cpu().numpy(), axis=1))
                    # print("Labels: ", np.argmax(y_batch.cpu().numpy(), axis=1))
                    val_accuracy.append(
                        np.mean(
                            np.argmax(y_pred.cpu().numpy(), axis=1)
                            == np.argmax(y_batch.cpu().numpy(), axis=1)
                        )
                    )

        return np.mean(val_loss), np.mean(val_accuracy)

    # write the training loop
    for epoch in tqdm(range(args.epochs)):

        # train for one epoch
        train_loss = train(args, model, device, train_loader, optimizer, criterion)

        # evaluate on validation set
        val_loss, val_accuracy = validate(args, model, device, val_loader, criterion)

        # remember best validation accuracy and save checkpoint
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)

        # print the results
        print(
            "Epoch: {}/{} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}".format(
                epoch + 1, args.epochs, train_loss, val_loss, val_accuracy,
            )
        )

    # evaluate on test set
    model.load_state_dict(torch.load(save_path))
    test_loss, test_accuracy = validate(args, model, device, test_loader, criterion)

    print("Test Loss: {:.4f} \tTest Accuracy: {:.4f}".format(test_loss, test_accuracy))
