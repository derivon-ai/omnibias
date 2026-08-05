# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sobolev preconditioning helpers (torch).

Generalised from an internal 2-D Navier-Stokes reference solver to
``D = 1, 2, 3`` spatial dimensions, decoupled from any specific PDE.

The Sobolev preconditioner downweights high-spatial-frequency residual
modes by a factor :math:`1 / (1 + |k|^4)^p`. ``p = 0`` reduces to plain
MSE. ``p = 1`` is the canonical biharmonic preconditioner -- this
(empirically and theoretically) accelerates training of stiff PDE
PINNs by *equalising* the gradient magnitude across resolved spatial
scales.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _spatial_k_grid_torch(
    n: int, L: float, *, device, dtype,
) -> Tensor:
    """1D wavenumber grid for an FFT of length ``n`` over period ``L``.

    We allocate ``fftfreq`` directly in ``dtype`` to avoid the
    float32-default precision loss before casting.
    """
    return 2.0 * math.pi * torch.fft.fftfreq(
        n, d=L / n, dtype=dtype, device=device,
    )


def _resolve_spatial_axes(
    residual: Tensor, spatial_axes: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if spatial_axes is not None:
        return tuple(int(a) for a in spatial_axes)
    if residual.dim() <= 1:
        raise ValueError(
            "sobolev_residual_loss needs at least 2 axes (time + spatial); "
            f"got shape {tuple(residual.shape)}"
        )
    return tuple(range(1, residual.dim()))


def sobolev_weight(
    residual: Tensor,
    *,
    L: float | tuple[float, ...],
    sobolev_p: float,
    spatial_axes: tuple[int, ...] | None = None,
) -> Tensor:
    """Build the per-mode preconditioning weight ``1 / (1 + |k|^4)^p``.

    Parameters
    ----------
    residual
        Tensor whose spatial axes will be FFT'd. Used only for shape
        and dtype.
    L
        Period of the spatial domain. Scalar broadcasts; tuple per axis.
    sobolev_p
        Smoothness exponent. ``p = 0`` -> constant ones;
        ``p = 1`` -> canonical biharmonic preconditioner.
    spatial_axes
        Axes of ``residual`` that are spatial. Default: all axes
        except axis 0 (time).
    """
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

    k2_total: Tensor | None = None
    for d, ax in enumerate(spatial_axes):
        k = _spatial_k_grid_torch(
            residual.shape[ax], L_t[d],
            device=residual.device, dtype=residual.dtype,
        )
        view_shape = [1] * residual.dim()
        view_shape[ax] = residual.shape[ax]
        k_view = k.view(*view_shape)
        contrib = k_view * k_view
        k2_total = contrib if k2_total is None else k2_total + contrib
    assert k2_total is not None
    stiffness = k2_total * k2_total                              # |k|^4
    return 1.0 / (1.0 + stiffness).pow(sobolev_p)


def sobolev_residual_loss(
    residual: Tensor,
    *,
    L: float | tuple[float, ...],
    sobolev_p: float,
    spatial_axes: tuple[int, ...] | None = None,
) -> Tensor:
    """Sobolev-preconditioned MSE on a gridded residual tensor.

    Computes :math:`\\mathrm{mean}_{t,\\,b} \\sum_k
    |\\hat R(k, t)|^2 / (1 + |k|^4)^p` where the sum is over spatial
    wavenumbers and the mean is over the *non-spatial* (typically:
    time and batch) axes. By Parseval this equals plain MSE when
    ``p = 0``.

    Parameters
    ----------
    residual
        Tensor of shape ``(..., n_1, ..., n_D)`` where the trailing
        axes are spatial.
    L
        Period(s) of the spatial domain.
    sobolev_p
        Smoothness exponent.
    spatial_axes
        Axes of ``residual`` that are spatial. Default: all-but-axis-0.
    """
    spatial_axes = _resolve_spatial_axes(residual, spatial_axes)
    if sobolev_p == 0:
        return (residual * residual).mean()

    R_hat = torch.fft.fftn(residual, dim=spatial_axes)
    n_spatial = 1
    for a in spatial_axes:
        n_spatial *= int(residual.shape[a])
    R_hat = R_hat / n_spatial
    weight = sobolev_weight(
        residual, L=L, sobolev_p=sobolev_p, spatial_axes=spatial_axes,
    )
    spectral = (R_hat.abs() ** 2 * weight).sum(dim=spatial_axes)
    return spectral.mean()


def mse_residual_loss(residual: Tensor) -> Tensor:
    """Plain MSE of a residual tensor (for non-gridded collocation)."""
    return (residual * residual).mean()


__all__ = [
    "mse_residual_loss",
    "sobolev_residual_loss",
    "sobolev_weight",
]
