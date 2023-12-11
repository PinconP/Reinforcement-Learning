import torch as th


def get_activation_function(name):
    """
    Maps a string name to a PyTorch activation function.

    Args:
    name (str): The name of the activation function as a string.

    Returns:
    torch.nn.Module: The corresponding PyTorch activation function.
    """
    activation_functions = {
        "relu": th.nn.ReLU,
        "tanh": th.nn.Tanh,
        "sigmoid": th.nn.Sigmoid,
        "leaky_relu": th.nn.LeakyReLU,
        "elu": th.nn.ELU,
        "softmax": th.nn.Softmax,
        # Add more mappings as needed
    }

    # Get the function from the dictionary, default to ReLU if not found
    return activation_functions.get(name.lower(), th.nn.ReLU)
