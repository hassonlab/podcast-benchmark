import torch

def to_tensor(array):
    return torch.from_numpy(array).float()