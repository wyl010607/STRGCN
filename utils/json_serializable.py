import json

import numpy as np
import torch

def json_serializable(obj):
    """
    Recursively makes an object JSON serializable by handling various data types,
    especially from NumPy and PyTorch that are not natively serializable by JSON.

    Parameters:
        obj (any): The object to make JSON serializable.

    Returns:
        any: The modified object that is JSON serializable.
    """
    # Handle lists and tuples recursively
    if isinstance(obj, (list, tuple)):
        return [json_serializable(o) for o in obj]

    # Handle dictionaries recursively
    elif isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}

    # Handle PyTorch tensors by converting to a list
    elif isinstance(obj, torch.Tensor):
        # Check if the tensor is on GPU, move to CPU first
        if obj.is_cuda:
            obj = obj.cpu()
        return obj.detach().numpy().tolist()

    # Handle various NumPy data types
    elif isinstance(obj, np.number):
        return obj.item()  # Convert numpy numbers to native Python types

    # Handle potential numpy arrays embedded in the data structure
    elif isinstance(obj, np.ndarray):
        # Convert arrays to list, handling the dtype conversion implicitly
        return obj.tolist()

    else:
        try:
            json.dumps(obj)
        except (TypeError, OverflowError):
            return str(obj)
        return obj
