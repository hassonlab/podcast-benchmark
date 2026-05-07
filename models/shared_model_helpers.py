import torch


def apply_activation(x, activation):
    if activation == "linear":
        return x
    elif activation == "tanh":
        return torch.tanh(x)
    elif activation == "relu":
        return torch.relu(x)
    elif activation == "sigmoid":
        return torch.sigmoid(x)
    elif activation == "softmax":
        return torch.softmax(x, dim=-1)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
