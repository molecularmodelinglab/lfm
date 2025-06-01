from common.utils import CONFIG
import torch
import torch.nn as nn
import torch.nn.functional as F


def rbf_encode(t, start, end, steps):
    """Encode Gaussian radial basis functions of the items in t."""
    mu = torch.linspace(start, end, steps, device=t.device)
    sigma = (start - end) / steps
    dists_expanded = t.unsqueeze(-1)
    mu_expanded = mu.view(*[1 for i in range(t.dim())], -1)
    diff = ((dists_expanded - mu_expanded) / sigma) ** 2
    return torch.exp(-diff)


def create_grid(spacing, n_channels, total_size):
    """Creates a grid for the receptor with spacing in angstroms"""
    res = int(total_size / spacing)

    # ret = CubicBSplineGrid3d(resolution=(res, res, res), n_channels=n_channels)
    # ret.data = torch.randn_like(ret.data)

    ret = torch.randn(n_channels, res, res, res)

    return ret


def make_linear(n_in, n_out):
    return nn.Sequential(
        nn.Linear(n_in, n_out),
        nn.LayerNorm(n_out),
        nn.SiLU(),
    )
