import torch
from torch import nn


def avg_l1_norm(x: torch.Tensor, eps=1e-8):
	return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def shrink_and_perturb(x: nn.Module, shrinkage: float=0.8, perturb: float=0.2) -> None:
    """
    Shrink and perturb the parameters of a module.

    Args:
        x: Module.
        shrinkage: Shrinkage factor.
        perturb: Perturbation factor.
    """
    with torch.no_grad():
        for p in x.parameters():
            p.mul_(shrinkage)
            p.add_(torch.randn_like(p) * perturb)


def weight_reset(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
    


def get_weight_decay_params(model: nn.Module) -> set:
    all_params = set(model.parameters())
    wd_params = set()
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            wd_params.add(m.weight)
    no_wd_params = all_params - wd_params

    return wd_params, no_wd_params


def gaussian_mse_loss(mu, logvar, target, logvar_loss = True):
    if len(mu.shape) != len(target.shape):
        print("gaussian_mse_loss: target shape not matching", mu.shape, target.shape)
    inv_var = (-logvar).exp()

    if logvar_loss:
        return (logvar + (target - mu)**2 * inv_var).mean()
    else:
        return ((target - mu)**2).mean()