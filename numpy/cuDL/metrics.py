import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def get_metric(name):
    name = name.lower()
    if name == "mse":
        return mse
    elif name == "rmse":
        return rmse
    elif name == "mae":
        return mae
    elif name == "accuracy":
        return accuracy
    elif name == "precision":
        return precision
    elif name == "recall":
        return recall
    elif name == "f1":
        return f1
    else:
        raise ValueError(f"Unknown metric: {name}")


# all popular metrics for evaulating regression and classification models

# regression metrics
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# classification metrics
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100


def precision(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100


def recall(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100


def f1(y_true, y_pred):
    return (
        2
        * precision(y_true, y_pred)
        * recall(y_true, y_pred)
        / (precision(y_true, y_pred) + recall(y_true, y_pred))
    )


def confusion_matrix(y_true, y_pred, labels=None):
    return confusion_matrix(y_true, y_pred, labels=labels)


def classification_report(y_true, y_pred, labels=None):
    return classification_report(y_true, y_pred, labels=labels)
