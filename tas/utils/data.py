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


def tensorify(x: Optional[ArrayLike], dtype=torch.float32, device=None, non_blocking=True) -> torch.Tensor:
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
        return x.to(device=device, dtype=dtype, non_blocking=non_blocking)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype, non_blocking=non_blocking)
        #return torch.tensor(x, dtype=dtype, device=device)
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


def seq_to_torch(x: T, dtype=torch.float32, device=None, non_blocking=True) -> T:
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
    return x.__class__(*[tensorify(getattr(x, k.name), dtype=dtype, device=device, non_blocking=non_blocking) for k in fields(x)])


def split_trajectory(traj: Trajectory, portion: float = 0.5) -> Tuple[Trajectory, Trajectory]:
    l = len(traj)
    split = int(l * portion)

    def _split1(name, v):
        if v is None:
            return None
        if name == 'all_observations':
            return v[:split + 1]
        return v[:split]


    def _split2(name, v):
        if v is None:
            return None
        return v[split:]


    return (
        Trajectory(**{k.name: _split1(k.name, getattr(traj, k.name)) for k in fields(traj)}),
        Trajectory(**{k.name: _split2(k.name, getattr(traj, k.name)) for k in fields(traj)}),
    )


def stack_trajectory(trajs: Sequence[Trajectory]) -> Trajectory:
    # Check if all trajs have the same length? fill with NaN if Trajectories are not consistent?
    # Just stick with easiest implementation for now
    return Trajectory(**{
        k.name: torch.stack([getattr(traj, k.name) for traj in trajs], dim=1)\
              for k in fields(trajs[0]) if getattr(trajs[0], k.name) is not None
    })