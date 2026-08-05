# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sobolev preconditioning helpers (jax twin of the torch module)."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array


def _spatial_k_grid_jax(n: int, L: float, dtype) -> Array:
    """1D wavenumber grid for an FFT of length ``n`` over period ``L``."""
    return (2.0 * math.pi) * jnp.fft.fftfreq(n, d=L / n).astype(dtype)


def _resolve_spatial_axes(
    residual: Array, spatial_axes: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if spatial_axes is not None:
        return tuple(int(a) for a in spatial_axes)
    if residual.ndim <= 1:
        raise ValueError(
            "sobolev_residual_loss needs at least 2 axes (time + spatial); "
            f"got shape {tuple(residual.shape)}"
        )
    return tuple(range(1, residual.ndim))


def sobolev_weight(
    residual: Array,
    *,
    L: float | tuple[float, ...],
    sobolev_p: float,
    spatial_axes: tuple[int, ...] | None = None,
) -> Array:
    """Build the per-mode preconditioning weight ``1 / (1 + |k|^4)^p``."""
    spatial_axes = _resolve_spatial_axes(residual, spatial_axes)
    if isinstance(L, int | float):
        L_t: tuple[float, ...] = tuple(float(L) for _ in spatial_axes)
    else:
        L_t = tuple(float(x) for x in L)
        if len(L_t) != len(spatial_axes):
            raise ValueError(
                f"L tuple length {len(L_t)} does not match spatial_axes "
                f"length {len(spatial_axes)}"
            )

    real_dtype = jnp.zeros((), dtype=residual.dtype).real.dtype
    k2_total: Array | None = None
    for d, ax in enumerate(spatial_axes):
        k = _spatial_k_grid_jax(int(residual.shape[ax]), L_t[d], real_dtype)
        view_shape = [1] * residual.ndim
        view_shape[ax] = int(residual.shape[ax])
        k_view = k.reshape(view_shape)
        contrib = k_view * k_view
        k2_total = contrib if k2_total is None else k2_total + contrib
    assert k2_total is not None
    stiffness = k2_total * k2_total                              # |k|^4
    return 1.0 / jnp.power(1.0 + stiffness, sobolev_p)


def sobolev_residual_loss(
    residual: Array,
    *,
    L: float | tuple[float, ...],
    sobolev_p: float,
    spatial_axes: tuple[int, ...] | None = None,
) -> Array:
    """Sobolev-preconditioned MSE on a gridded residual tensor (jax)."""
    spatial_axes = _resolve_spatial_axes(residual, spatial_axes)
    if sobolev_p == 0:
        return jnp.mean(residual * residual)

    R_hat = jnp.fft.fftn(residual, axes=spatial_axes)
    n_spatial = 1
    for a in spatial_axes:
        n_spatial *= int(residual.shape[a])
    R_hat = R_hat / n_spatial
    weight = sobolev_weight(
        residual, L=L, sobolev_p=sobolev_p, spatial_axes=spatial_axes,
    )
    spectral = jnp.sum(jnp.abs(R_hat) ** 2 * weight, axis=spatial_axes)
    return jnp.mean(spectral)


def mse_residual_loss(residual: Array) -> Array:
    """Plain MSE of a residual tensor (jax twin)."""
    return jnp.mean(residual * residual)


__all__ = [
    "mse_residual_loss",
    "sobolev_residual_loss",
    "sobolev_weight",
]
