# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""High-accuracy numerical whole-line Hilbert transform (torch).

Torch twin of :mod:`omnibias.pinn.jax.hilbert_line`. Same quadrature
contract: split-core subtracted-kernel GL plus a power-mapped algebraic
tail. Numerical, not a closed-form Hardy transform and not a certificate.

Honesty: planted ``H[Q]=-P`` can sit near ``1e-14``. Stretch ``1e-13`` is
still unearned on a trained Wang net.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from torch import Tensor

_GL_PM1: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_GL_01: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gl_pm1(n: int, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    n = int(n)
    cached = _GL_PM1.get(n)
    if cached is None:
        xi, w = leggauss(n)
        cached = (xi.astype(np.float64), w.astype(np.float64))
        _GL_PM1[n] = cached
    return (
        torch.as_tensor(cached[0], dtype=dtype, device=device),
        torch.as_tensor(cached[1], dtype=dtype, device=device),
    )


def _gl_01(n: int, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    n = int(n)
    cached = _GL_01.get(n)
    if cached is None:
        xi, w = leggauss(n)
        u = np.clip(0.5 * (xi + 1.0), 1e-15, 1.0)
        cached = (u.astype(np.float64), (0.5 * w).astype(np.float64))
        _GL_01[n] = cached
    return (
        torch.as_tensor(cached[0], dtype=dtype, device=device),
        torch.as_tensor(cached[1], dtype=dtype, device=device),
    )


def hilbert_gl_panel(
    y: Tensor,
    values: Tensor,
    omega_fn: Callable[[Tensor], Tensor],
    lo: float,
    hi: float,
    n_gl: int,
) -> Tensor:
    """Subtracted-kernel GL integral of ``(Ω(t)-Ω(x))/(x-t)`` on ``[lo, hi]``."""
    y = torch.as_tensor(y).reshape(-1).to(dtype=torch.float64)
    values = torch.as_tensor(values).reshape(-1).to(dtype=torch.float64, device=y.device)
    xi, w = _gl_pm1(n_gl, device=y.device, dtype=y.dtype)
    mid = 0.5 * (float(lo) + float(hi))
    half = 0.5 * (float(hi) - float(lo))
    t = mid + half * xi
    f_t = torch.as_tensor(omega_fn(t), dtype=torch.float64, device=y.device).reshape(-1)
    dt = y.unsqueeze(1) - t.unsqueeze(0)
    near = dt.abs() < 1e-14
    dt_safe = torch.where(near, torch.ones_like(dt), dt)
    kern = torch.where(near, torch.zeros_like(dt), (f_t.unsqueeze(0) - values.unsqueeze(1)) / dt_safe)
    return half * (kern * w.unsqueeze(0)).sum(dim=1)


def hilbert_gl_split_core(
    y: Tensor,
    values: Tensor,
    omega_fn: Callable[[Tensor], Tensor],
    *,
    y_trunc: float,
    y_near: float,
    n_near: int,
    n_far: int,
) -> Tensor:
    """Finite-interval PV Hilbert via three GL panels plus the log endpoint term."""
    y = torch.as_tensor(y).reshape(-1).to(dtype=torch.float64)
    values = torch.as_tensor(values).reshape(-1).to(dtype=torch.float64, device=y.device)
    Y = float(y_trunc)
    yn = min(float(y_near), 0.5 * Y)
    if Y <= 0.0 or yn <= 0.0:
        raise ValueError(f"y_trunc and y_near must be > 0, got {Y}, {yn}")
    integ = (
        hilbert_gl_panel(y, values, omega_fn, -Y, -yn, n_far)
        + hilbert_gl_panel(y, values, omega_fn, -yn, yn, n_near)
        + hilbert_gl_panel(y, values, omega_fn, yn, Y, n_far)
    )
    log_term = values * torch.log(
        torch.clamp(y + Y, min=1e-12) / torch.clamp(Y - y, min=1e-12)
    )
    return (integ + log_term) / math.pi


def hilbert_tail_power_mapped(
    y: Tensor,
    omega_fn: Callable[[Tensor], Tensor],
    *,
    y_trunc: float,
    decay_power: float,
    n_tail: int,
) -> Tensor:
    r"""Mapped ``|t|>Y`` tail for odd ``Ω`` with algebraic decay ``|t|^{-α}``."""
    y = torch.as_tensor(y).reshape(-1).to(dtype=torch.float64)
    Y = float(y_trunc)
    alpha = float(decay_power)
    if Y <= 0.0:
        raise ValueError(f"y_trunc must be > 0, got {Y}")
    if not (alpha > 0.0):
        raise ValueError(f"decay_power must be > 0, got {alpha}")
    v, w = _gl_01(n_tail, device=y.device, dtype=y.dtype)
    u = torch.pow(v, 1.0 / alpha)
    t_pos = Y / torch.clamp(u, min=1e-15, max=1.0)
    omega_pos = torch.as_tensor(omega_fn(t_pos), dtype=torch.float64, device=y.device).reshape(-1)
    g = omega_pos * torch.pow(t_pos, alpha)
    xi = torch.clamp(y / Y, min=-1.0 + 1e-8, max=1.0 - 1e-8)
    den = torch.clamp(1.0 - (xi.unsqueeze(1) ** 2) * (u.unsqueeze(0) ** 2), min=1e-18)
    integ = ((g.unsqueeze(0) / den) * w.unsqueeze(0)).sum(dim=1) / alpha
    integ = integ * (Y ** (-alpha))
    return (-2.0 / math.pi) * integ


def hilbert_gl_single_raw_tail(
    y: Tensor,
    values: Tensor,
    omega_fn: Callable[[Tensor], Tensor],
    *,
    y_trunc: float,
    n_gl: int = 96,
    n_tail: int = 48,
) -> Tensor:
    """Single-panel GL core + raw-``u`` tail (campaign hop14 baseline)."""
    y = torch.as_tensor(y).reshape(-1).to(dtype=torch.float64)
    values = torch.as_tensor(values).reshape(-1).to(dtype=torch.float64, device=y.device)
    Y = float(y_trunc)
    integ = hilbert_gl_panel(y, values, omega_fn, -Y, Y, int(n_gl))
    log_term = values * torch.log(
        torch.clamp(y + Y, min=1e-12) / torch.clamp(Y - y, min=1e-12)
    )
    h_core = (integ + log_term) / math.pi
    u01, w01 = _gl_01(int(n_tail), device=y.device, dtype=y.dtype)
    t_pos = Y / torch.clamp(u01, min=1e-8, max=1.0)
    omega_pos = torch.as_tensor(omega_fn(t_pos), dtype=torch.float64, device=y.device).reshape(
        -1
    )
    xi_y = torch.clamp(y / Y, min=-1.0 + 1e-8, max=1.0 - 1e-8)
    den = torch.clamp(1.0 - (xi_y.unsqueeze(1) ** 2) * (u01.unsqueeze(0) ** 2), min=1e-18)
    h_tail = (-2.0 / math.pi) * (((omega_pos / u01) / den) * w01.unsqueeze(0)).sum(dim=1)
    return h_core + h_tail


def hilbert_wholeline_hp(
    y: Tensor,
    values: Tensor,
    omega_fn: Callable[[Tensor], Tensor],
    *,
    decay_power: float,
    y_trunc: float | None = None,
    y_near: float = 2.0,
    n_near: int = 128,
    n_far: int = 64,
    n_tail: int = 96,
) -> Tensor:
    """Split-core + power-mapped-tail whole-line Hilbert of a free odd ``Ω``."""
    y = torch.as_tensor(y).reshape(-1).to(dtype=torch.float64)
    values = torch.as_tensor(values).reshape(-1).to(dtype=torch.float64, device=y.device)
    Y = float(y_trunc) if y_trunc is not None else float(torch.max(torch.abs(y)))
    h_core = hilbert_gl_split_core(
        y,
        values,
        omega_fn,
        y_trunc=Y,
        y_near=y_near,
        n_near=int(n_near),
        n_far=int(n_far),
    )
    h_tail = hilbert_tail_power_mapped(
        y,
        omega_fn,
        y_trunc=Y,
        decay_power=float(decay_power),
        n_tail=int(n_tail),
    )
    return h_core + h_tail


__all__ = [
    "hilbert_gl_panel",
    "hilbert_gl_single_raw_tail",
    "hilbert_gl_split_core",
    "hilbert_tail_power_mapped",
    "hilbert_wholeline_hp",
]
