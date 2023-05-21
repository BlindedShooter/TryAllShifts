from typing import Sequence

import torch
import torch.nn as nn



def construct_mlp(arch: Sequence[int], activation: torch.nn.Module,
                  last_activation: torch.nn.Module=nn.Identity) -> torch.nn.Module:
    """
    Constructs a Sequential MLP with given architecture and activation.
    Beware that this function does not apply activation on the final layer.

    params:
        - arch: an iterable of int, which is the hidden sizes of the MLP.
        - activation: a non-linear activation layer class.

    returns:
        - A torch.nn.Sequential MLP model
    """
    layers = []

    for inp, out in zip(arch[:-1], arch[1:]):
        layers.append(nn.Linear(inp, out))
        layers.append(activation())
    layers.pop(-1)  # removes last activation
    layers.append(last_activation())

    return nn.Sequential(*layers)
