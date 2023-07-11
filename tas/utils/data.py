from typing import TypeVar, Iterable, Generator
from dataclasses import fields
import numpy as np
import torch

from tas.typing import *

T = TypeVar('T')


def infinite_generator(x: Iterable[T]) -> Generator[T, None, None]:
    """
    Infinite generator from an iterable.
    
    Args:
        x: Iterable.
    Returns:
        Infinite generator.
    """
    while True:
        for v in x:
            yield v


def numpify(x: Optional[ArrayLike]) -> np.ndarray:
    """
    Convert ArrayLike to numpy array.

    Args:
        x: ArrayLike object.
    Returns:
        Numpy array.
    """
    if x is None:
        return None
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (list, tuple)):
        return np.array(x)
    else:
        #raise ValueError(f'Unknown type: {type(x)}')
        return x


def tensorify(x: Optional[ArrayLike], dtype=torch.float32, device=None) -> torch.Tensor:
    """
    Convert ArrayLike to torch tensor.
    
    Args:
        x: ArrayLike object.
        dtype: Data type of the returned tensor.
        device: Device of the returned tensor.
        copy: Whether to copy the data.
    Returns:
        Torch tensor.
    """
    if x is None:
        return None
    elif isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    elif isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=dtype, device=device)
    else:
        #raise ValueError(f'Unknown type: {type(x)}')
        return x


def seq_to_numpy(x: T) -> T:
    """
    Convert a Dataclass of ArrayLike objects as numpy arrays.
    
    Args:
        x: Sequence of ArrayLike objects.
    Returns:
        Sequence of numpy arrays."""
    return x.__class__(*[numpify(getattr(x, k.name)) for k in fields(x)])


def seq_to_torch(x: T, dtype=torch.float32, device=None) -> T:
    """
    Convert a Dataclass of ArrayLike objects as torch tensors.

    Args:
        x: Sequence of ArrayLike objects.
        dtype: Data type of the returned tensors.
        device: Device of the returned tensors.
        copy: Whether to copy the data.
    Returns:
        Sequence of torch tensors.
    """
    return x.__class__(*[tensorify(getattr(x, k.name), dtype=dtype, device=device) for k in fields(x)])
