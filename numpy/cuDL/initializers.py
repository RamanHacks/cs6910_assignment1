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
    return weights.astype(np.float32)


def zeros_init(*shape):
    return np.zeros(*shape).astype(np.float32)


def ones_init(*shape):
    return np.ones(*shape).astype(np.float32)


def xavier_init(*shape):
    # glorot normal

    assert len(shape) == 2, "xavier_init only supports 2D tensors"
    in_dim, out_dim = shape[0], shape[1]

    scale = np.sqrt(2 / (in_dim + out_dim))
    mu = 0

    # return np.random.randn(*shape).astype(np.float32) * scale.astype(np.float32)
    return np.random.normal(mu, scale, size=shape).astype(np.float32)


def he_init(*shape):
    assert len(shape) == 2, "he_init only supports 2D tensors"
    return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / shape[0]).astype(
        np.float32
    )
