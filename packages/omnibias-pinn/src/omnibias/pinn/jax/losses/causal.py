# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wang & Perdikaris causal weighting (jax twin)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from omnibias.pinn.jax.losses.sobolev import (
    _resolve_spatial_axes,
    sobolev_weight,
)


@dataclass(frozen=True)
class CausalConfig:
    """Configuration for Wang & Perdikaris causal weighting."""

    enabled: bool = False
    epsilon: float = 1.0
    n_time_bins: int = 32


def causal_weights_from_per_bin(L_per_bin: Array, *, epsilon: float) -> Array:
    """``w_i = exp(-epsilon * sum_{j<i} L_j_stop_grad)``.

    Returns a non-increasing weight vector with ``stop_gradient``
    applied (so causal weights do not contribute to the gradient
    signal -- only the per-bin loss does).
    """
    if L_per_bin.ndim != 1:
        raise ValueError(
            f"expected 1D per-bin losses, got shape {tuple(L_per_bin.shape)}"
        )
    import jax  # local to keep top-level import light
    L_det = jax.lax.stop_gradient(L_per_bin)
    cum = jnp.cumsum(L_det, axis=0)
    cum_lt = jnp.concatenate([jnp.zeros_like(cum[:1]), cum[:-1]], axis=0)
    return jnp.exp(-epsilon * cum_lt)


def causal_residual_loss(
    residual_t_first: Array,
    *,
    epsilon: float,
    L: float | tuple[float, ...] | None = None,
    sobolev_p: float = 0.0,
    spatial_axes: tuple[int, ...] | None = None,
    return_weights: bool = False,
) -> Array | tuple[Array, Array]:
    """Causal-weighted (and optionally Sobolev-preconditioned) MSE."""
    if residual_t_first.ndim < 2:
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
        R_hat = jnp.fft.fftn(residual_t_first, axes=spatial_axes)
        n_spatial = 1
        for a in spatial_axes:
            n_spatial *= int(residual_t_first.shape[a])
        R_hat = R_hat / n_spatial
        weight = sobolev_weight(
            residual_t_first, L=L, sobolev_p=sobolev_p,
            spatial_axes=spatial_axes,
        )
        L_per_bin = jnp.sum(jnp.abs(R_hat) ** 2 * weight, axis=spatial_axes)
    else:
        L_per_bin = jnp.mean(residual_t_first ** 2, axis=spatial_axes)
    if L_per_bin.ndim != 1:
        raise ValueError(
            "causal_residual_loss expects exactly one time axis; reduced "
            f"per-bin loss has shape {tuple(L_per_bin.shape)}"
        )
    w = causal_weights_from_per_bin(L_per_bin, epsilon=epsilon)
    loss = jnp.mean(w * L_per_bin)
    if return_weights:
        return loss, w
    return loss


__all__ = [
    "CausalConfig",
    "causal_residual_loss",
    "causal_weights_from_per_bin",
]
