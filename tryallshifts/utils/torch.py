from typing import Iterable

import torch



def shrink_and_perturb(
    params: Iterable[torch.Tensor],
    shrink: float = 0.8,
    perturb: float = 0.2
) -> None:
    """
    Implements Shrink & Perturb on Pytorch parameters.
    The operation in done in-place.
    The sum of parameters shrink and perturb does not have to be equal to 1.

    Usage:
        shrink_and_perturb(model.parameters(), shrink=0.8, perturb=0.2)

    - params
        - params: torch parameters to apply shrink and perturb. the operation is done in-place.
        - shrink: How much the model weight is shrinked to. 
                 The value is directly multiplied on the model weight.
        - perturb: How much the Gaussian noise will be added to model weight. 
    """
    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param in params:
            param.data.mul_(shrink)
            torch.add(param.data, torch.randn_like(param.data), alpha=perturb, out=param.data)
