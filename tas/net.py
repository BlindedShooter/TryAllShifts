import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tas.typing import *
from tas.utils import *



class PlainMLP(nn.Module):
    """
    A plain MLP with ReLU default activation.
    
    Args:
        inp_dim: Input dimension.
        out_dim: Output dimension.
        arch: Architecture of the MLP.
        actv: Activation function.
        out_actv: Activation function for the output layer.
        layer_norm: Whether to use layer normalization.
        bias: Whether to use bias.
    """
    def __init__(self, inp_dim: int, out_dim: int, arch: Sequence[int], actv: nn.Module=nn.ReLU,
                 out_actv: Optional[nn.Module]=nn.Identity, layer_norm: bool=False, bias: bool=True, **_) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i, (inp, out) in enumerate(zip([inp_dim] + arch[:-1], arch)):
            self.layers.append(nn.Linear(inp, out, bias=bias))
            self.layers.append(getattr(nn, actv)())
            if layer_norm and i != len(arch):  # no layer norm for the last layer
                self.layers.append(nn.LayerNorm(out))
        self.layers.append(nn.Linear(arch[-1], out_dim, bias=bias))
        self.layers.append(getattr(nn, out_actv)())

        self.layers = nn.Sequential(*self.layers)


    def forward(self, x: torch.Tensor, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.layers(x)
    


class ConcatFirstMLP(nn.Module):
    """
    A MLP that concatenates the input with a context vector at first.

    Args:
        inp_dim: Input dimension.
        context_dim: Context dimension.
        out_dim: Output dimension.
        arch: Architecture of the MLP.
        actv: Activation function.
        out_actv: Activation function for the output layer.
        layer_norm: Whether to use layer normalization.
        bias: Whether to use bias.
    """
    def __init__(self, inp_dim: int, context_dim: int, out_dim: int, arch: Sequence[int], actv: str='ReLU',
                 out_actv: str='Identity', layer_norm: bool=False, bias: bool=True, **_) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i, (inp, out) in enumerate(zip([inp_dim + context_dim] + arch[:-1], arch)):
            self.layers.append(nn.Linear(inp, out, bias=bias))
            self.layers.append(getattr(nn, actv)())
            if layer_norm and i != len(arch) - 1:
                self.layers.append(nn.LayerNorm(out))
        self.layers.append(nn.Linear(arch[-1], out_dim, bias=bias))
        self.layers.append(getattr(nn, out_actv)())

        self.layers = nn.Sequential(*self.layers)


    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat([x, c], dim=-1))



NN_TYPE_TO_CLS = {
    "plain": PlainMLP,
    #"gated": GatedMLP,
    "concat_first": ConcatFirstMLP,
    #"concat_last": ConcatLastMLP,
    #"concat_full": ConcatFullMLP,
    #"bilinear_first": BilinearFirstMLP,
    #"bilinear_last": BilinearLastMLP,
    #"bilinear_full": BilinearFullMLP,
    #"torch_bilinear_first": TorchBilinearFirstMLP,
    #"torch_bilinear_last": TorchBilinearLastMLP,
    #"torch_bilinear_full": TorchBilinearFullMLP,
    #"film_full": FilmFullMLP,
    #"film_first": FilmFirstMLP,
    #"film_last": FilmLastMLP,
}