# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wang & Perdikaris causal weighting (torch).

Generalised from an internal 2-D Navier-Stokes reference solver and
made equation-agnostic: the helpers consume a ``(n_t, ...spatial)``
residual tensor and return a (causal-weighted, optionally
Sobolev-preconditioned) scalar loss.

Reference
---------
Wang, Sankaran & Perdikaris, *Respecting Causality is All You Need for
Training Physics-informed Neural Networks*, arXiv:2203.07404 (2022).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from omnibias.pinn.torch.losses.sobolev import (
    _resolve_spatial_axes,
    sobolev_weight,
)
from torch import Tensor


@dataclass(frozen=True)
class CausalConfig:
    """Configuration for Wang & Perdikaris causal weighting."""

    enabled: bool = False
    epsilon: float = 1.0
    n_time_bins: int = 32


def causal_weights_from_per_bin(
    L_per_bin: Tensor, *, epsilon: float,
) -> Tensor:
    """Return ``w_i = exp(-epsilon * sum_{j<i} L_j_detached)``.

    The weights are non-increasing and detached from autograd.
    """
    if L_per_bin.dim() != 1:
        raise ValueError(
            f"expected 1D per-bin losses, got shape {tuple(L_per_bin.shape)}"
        )
    L_det = L_per_bin.detach()
    cum = torch.cumsum(L_det, dim=0)
    cum_lt = torch.cat([torch.zeros_like(cum[:1]), cum[:-1]], dim=0)
    return torch.exp(-epsilon * cum_lt)


def causal_residual_loss(
    residual_t_first: Tensor,
    *,
    epsilon: float,
    L: float | tuple[float, ...] | None = None,
    sobolev_p: float = 0.0,
    spatial_axes: tuple[int, ...] | None = None,
    return_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Causal-weighted (and optionally Sobolev-preconditioned) MSE.

    The leading axis of ``residual_t_first`` is interpreted as time
    (sorted in increasing order). Spatial axes follow.

    Parameters
    ----------
    residual_t_first
        Tensor of shape ``(n_t, n_1, ..., n_D)``.
    epsilon
        Causal sharpness parameter.
    L
        Period(s) of the spatial domain. Required when ``sobolev_p > 0``.
    sobolev_p
        Sobolev exponent for the per-bin loss. ``0`` -> plain MSE per
        bin.
    spatial_axes
        Optional override; defaults to all-but-axis-0.
    return_weights
        If true, also return the weights tensor.
    """
    if residual_t_first.dim() < 2:
        raise ValueError(
            f"causal_residual_loss expects ``(n_t, ...spatial)``; got "
            f"shape {tuple(residual_t_first.shape)}"
        )
    spatial_axes = _resolve_spatial_axes(residual_t_first, spatial_axes)
    if sobolev_p > 0:
        if L is None:
            raise ValueError(
                "causal_residual_loss with sobolev_p > 0 requires L"
            )
        R_hat = torch.fft.fftn(residual_t_first, dim=spatial_axes)
        n_spatial = 1
        for a in spatial_axes:
            n_spatial *= int(residual_t_first.shape[a])
        R_hat = R_hat / n_spatial
        weight = sobolev_weight(
            residual_t_first, L=L, sobolev_p=sobolev_p,
            spatial_axes=spatial_axes,
        )
        L_per_bin = (R_hat.abs() ** 2 * weight).sum(dim=spatial_axes)
    else:
        L_per_bin = (residual_t_first ** 2).mean(dim=spatial_axes)
    if L_per_bin.dim() != 1:
        raise ValueError(
            "causal_residual_loss expects exactly one time axis; reduced "
            f"per-bin loss has shape {tuple(L_per_bin.shape)}"
        )
    w = causal_weights_from_per_bin(L_per_bin, epsilon=epsilon)
    loss = (w * L_per_bin).mean()
    if return_weights:
        return loss, w
    return loss


__all__ = [
    "CausalConfig",
    "causal_residual_loss",
    "causal_weights_from_per_bin",
]
