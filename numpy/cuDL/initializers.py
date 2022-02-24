import numpy as np

# we could write this with Initialization class but feels like an overkill for now


def get_initializer(name):
    name = name.lower()
    if name == "random":
        return random_init
    elif name == "zeros":
        return zeros_init
    elif name == "ones":
        return ones_init
    elif name == "xavier":
        return xavier_init
    elif name == "he":
        return he_init
    else:
        raise ValueError(f"Unknown initializer: {name}")


def random_init(*shape):
    return np.random.rand(*shape)


def zeros_init(*shape):
    return np.zeros(*shape)


def ones_init(*shape):
    return np.ones(*shape)


def xavier_init(*shape):
    assert len(shape) == 2, "xavier_init only supports 2D tensors"
    return np.random.randn(*shape) * np.sqrt(2.0 / (shape[0] + shape[1]))


def he_init(*shape):
    assert len(shape) == 2, "he_init only supports 2D tensors"
    return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
