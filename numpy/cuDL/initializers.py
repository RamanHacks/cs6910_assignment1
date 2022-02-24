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
    weights = np.random.rand(*shape)
    # normalize the weights
    weights = weights / np.linalg.norm(weights)
    return weights


def zeros_init(*shape):
    return np.zeros(*shape)


def ones_init(*shape):
    return np.ones(*shape)


def xavier_init(*shape):

    mu, sigma = 0, 0.1  # mean and standard deviation
    assert len(shape) == 2, "xavier_init only supports 2D tensors"
    in_dim, out_dim = shape[0], shape[1]
    return np.random.normal(mu, sigma, (in_dim, out_dim)) * np.sqrt(
        2.0 / (in_dim + out_dim)
    )


def he_init(*shape):
    assert len(shape) == 2, "he_init only supports 2D tensors"
    return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
