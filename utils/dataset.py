import torch
from torch.utils.data import Dataset


class NeuralDictDataset(Dataset):
    """
    A PyTorch Dataset that takes neural data, a dictionary of tensors as input, and a target tensor.

    Args:
        neural_data: Tensor containing neural data inputs.
        input_dict: Dictionary where keys are strings and values are tensors.
                   All tensors must have the same length in dimension 0.
        target: Target tensor with the same length as input tensors in dimension 0.
    """

    def __init__(self, neural_data, input_dict, target):
        self.neural_data = neural_data
        self.input_dict = input_dict
        self.target = target

        # Validate that all tensors have the same length
        lengths = [len(v) for v in input_dict.values()]
        if not all(length == len(target) for length in lengths):
            raise ValueError(
                "All input tensors and target must have the same length in dimension 0"
            )

        self.length = len(target)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return a tuple: (dict of ith indexed tensors, ith target)
        item_dict = {key: value[idx] for key, value in self.input_dict.items()}
        return self.neural_data[idx], item_dict, self.target[idx]
