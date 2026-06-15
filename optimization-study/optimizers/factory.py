import torch.optim as optim


def get_optimizer(name, model_params, lr=0.001):

    name = name.lower()

    if name == "sgd":
        return optim.SGD(
            model_params,
            lr=lr
        )

    elif name == "momentum":
        return optim.SGD(
            model_params,
            lr=lr,
            momentum=0.9
        )

    elif name == "adam":
        return optim.Adam(
            model_params,
            lr=lr
        )

    elif name == "adamw":
        return optim.AdamW(
            model_params,
            lr=lr
        )

    elif name == "rmsprop":
        return optim.RMSprop(
            model_params,
            lr=lr,
            alpha=0.99
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {name}"
        )