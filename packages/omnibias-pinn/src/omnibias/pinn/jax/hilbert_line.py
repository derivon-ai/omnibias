# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""High-accuracy numerical whole-line Hilbert transform (JAX).

The CCF stretch gate is ``1e-13``. Closed-form ``H`` exists only on a Hardy
span. A free neural ``Ω`` needs a quadrature Hilbert whose error is *far*
below that gate; periodic FFT and single-panel GL96 + raw-``u`` tail sit
near ``1e-3``–``1e-1`` on planted ``Q_{a,α}``.

This module is a **numerical** whole-line operator, not a certificate and
not a closed-form Hardy transform:

* **Core.** Split Gauss–Legendre on the subtracted PV kernel
  ``(Ω(t)-Ω(x))/(x-t)`` so nodes cluster on a near-field panel (CCF peaks
  live near the origin; a single GL panel on ``[-Y,Y]`` undersamples
  ``y~0``).
* **Tail.** The mapped integrand ``Ω(Y/u)/u`` behaves as ``u^{α-1}`` for
  ``Ω ~ |t|^{-α}``. Raw GL on ``u∈(0,1]`` is not spectral. The substitution
  ``u = v^{1/α}`` flattens that weight so GL in ``v`` is.

Honesty: float64 can store ``1e-13``; this quadrature still has to *earn*
it. Tests report the measured ``H[Q]+P`` floor. Stretch stays ``1e-13``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.polynomial.legendre import leggauss

jax.config.update("jax_enable_x64", True)


def _require_float64(name: str) -> None:
    """Stretch Hilbert must not silently run in float32 (JAX default)."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            f"{name} requires jax_enable_x64=True; float32 floors the 1e-13 Hilbert"
        )

_GL_PM1: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_GL_01: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gl_pm1(n: int) -> tuple[Array, Array]:
    n = int(n)
    cached = _GL_PM1.get(n)
    if cached is None:
        xi, w = leggauss(n)
        cached = (xi.astype(np.float64), w.astype(np.float64))
        _GL_PM1[n] = cached
    return (
        jnp.asarray(cached[0], dtype=jnp.float64),
        jnp.asarray(cached[1], dtype=jnp.float64),
    )


def _gl_01(n: int) -> tuple[Array, Array]:
    n = int(n)
    cached = _GL_01.get(n)
    if cached is None:
        xi, w = leggauss(n)
        u = np.clip(0.5 * (xi + 1.0), 1e-15, 1.0)
        cached = (u.astype(np.float64), (0.5 * w).astype(np.float64))
        _GL_01[n] = cached
    return (
        jnp.asarray(cached[0], dtype=jnp.float64),
        jnp.asarray(cached[1], dtype=jnp.float64),
    )


def hilbert_gl_panel(
    y: Array,
    values: Array,
    omega_fn,
    lo: float,
    hi: float,
    n_gl: int,
) -> Array:
    """Subtracted-kernel GL integral of ``(Ω(t)-Ω(x))/(x-t)`` on ``[lo, hi]``."""
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    values = jnp.asarray(values, dtype=jnp.float64).reshape(-1)
    xi, w = _gl_pm1(n_gl)
    mid = 0.5 * (float(lo) + float(hi))
    half = 0.5 * (float(hi) - float(lo))
    t = mid + half * xi
    f_t = jnp.asarray(omega_fn(t), dtype=jnp.float64).reshape(-1)
    dt = y[:, None] - t[None, :]
    near = jnp.abs(dt) < 1e-14
    dt_safe = jnp.where(near, jnp.ones_like(dt), dt)
    kern = jnp.where(near, 0.0, (f_t[None, :] - values[:, None]) / dt_safe)
    return half * (kern * w[None, :]).sum(axis=1)


def hilbert_gl_split_core(
    y: Array,
    values: Array,
    omega_fn,
    *,
    y_trunc: float,
    y_near: float,
    n_near: int,
    n_far: int,
) -> Array:
    """Finite-interval PV Hilbert via three GL panels plus the log endpoint term."""
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    values = jnp.asarray(values, dtype=jnp.float64).reshape(-1)
    Y = float(y_trunc)
    yn = min(float(y_near), 0.5 * Y)
    if Y <= 0.0 or yn <= 0.0:
        raise ValueError(f"y_trunc and y_near must be > 0, got {Y}, {yn}")
    integ = (
        hilbert_gl_panel(y, values, omega_fn, -Y, -yn, n_far)
        + hilbert_gl_panel(y, values, omega_fn, -yn, yn, n_near)
        + hilbert_gl_panel(y, values, omega_fn, yn, Y, n_far)
    )
    log_term = values * jnp.log(
        jnp.clip(y + Y, 1e-12) / jnp.clip(Y - y, 1e-12)
    )
    return (integ + log_term) / math.pi


def hilbert_tail_power_mapped(
    y: Array,
    omega_fn,
    *,
    y_trunc: float,
    decay_power: float,
    n_tail: int,
) -> Array:
    r"""Mapped ``|t|>Y`` tail for odd ``Ω`` with algebraic decay ``|t|^{-α}``.

    ``H_tail(x) = -(2/π) ∫_0^1 [Ω(Y/u)/u] / (1-ξ² u²) du``, ``ξ=x/Y``.
    For ``Ω ~ |t|^{-α}`` the integrand carries ``u^{α-1}``. The map
    ``u = v^{1/α}`` turns that weight into a constant so GL in ``v`` is
    spectral at the origin of the ``u``-chart.
    """
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    Y = float(y_trunc)
    alpha = float(decay_power)
    if Y <= 0.0:
        raise ValueError(f"y_trunc must be > 0, got {Y}")
    if not (alpha > 0.0):
        raise ValueError(f"decay_power must be > 0, got {alpha}")
    v, w = _gl_01(n_tail)
    u = jnp.power(v, 1.0 / alpha)
    t_pos = Y / jnp.clip(u, 1e-15, 1.0)
    omega_pos = jnp.asarray(omega_fn(t_pos), dtype=jnp.float64).reshape(-1)
    g = omega_pos * jnp.power(t_pos, alpha)
    xi = jnp.clip(y / Y, -1.0 + 1e-8, 1.0 - 1e-8)
    den = jnp.clip(1.0 - (xi[:, None] ** 2) * (u[None, :] ** 2), 1e-18)
    integ = ((g[None, :] / den) * w[None, :]).sum(axis=1) / alpha
    integ = integ * (Y ** (-alpha))
    return (-2.0 / math.pi) * integ


def hilbert_gl_single_raw_tail(
    y: Array,
    values: Array,
    omega_fn,
    *,
    y_trunc: float,
    n_gl: int = 96,
    n_tail: int = 48,
) -> Array:
    """Single-panel GL core + raw-``u`` tail (campaign hop14 baseline)."""
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    values = jnp.asarray(values, dtype=jnp.float64).reshape(-1)
    Y = float(y_trunc)
    integ = hilbert_gl_panel(y, values, omega_fn, -Y, Y, int(n_gl))
    log_term = values * jnp.log(
        jnp.clip(y + Y, 1e-12) / jnp.clip(Y - y, 1e-12)
    )
    h_core = (integ + log_term) / math.pi
    u01, w01 = _gl_01(int(n_tail))
    t_pos = Y / jnp.clip(u01, 1e-8, 1.0)
    omega_pos = jnp.asarray(omega_fn(t_pos), dtype=jnp.float64).reshape(-1)
    xi_y = jnp.clip(y / Y, -1.0 + 1e-8, 1.0 - 1e-8)
    den = jnp.clip(1.0 - (xi_y[:, None] ** 2) * (u01[None, :] ** 2), 1e-18)
    h_tail = (-2.0 / math.pi) * jnp.sum(
        ((omega_pos / u01) / den) * w01[None, :], axis=1
    )
    return h_core + h_tail


def hilbert_wholeline_hp(
    y: Array,
    values: Array,
    omega_fn,
    *,
    decay_power: float,
    y_trunc: float | None = None,
    y_near: float = 2.0,
    n_near: int = 128,
    n_far: int = 64,
    n_tail: int = 96,
) -> Array:
    """Split-core + power-mapped-tail whole-line Hilbert of a free odd ``Ω``.

    Parameters
    ----------
    y, values
        Sample locations and ``Ω(y)``. ``omega_fn`` must reproduce ``values``
        at ``y`` and be evaluable off-grid (tails and GL nodes).
    decay_power
        Far-field exponent ``α`` in ``Ω ~ |y|^{-α}`` (CCF: ``1/(1+λ)``).
    y_trunc
        Interior cutoff ``Y``. Defaults to ``max|y|``.
    y_near
        Half-width of the origin-centered GL panel.
    n_near, n_far, n_tail
        GL counts for the near panel, each outer panel, and the ``v``-tail.

    Returns
    -------
    Array
        Numerical ``HΩ`` at ``y``. Not a Hardy closed form.
    """
    _require_float64("hilbert_wholeline_hp")
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    values = jnp.asarray(values, dtype=jnp.float64).reshape(-1)
    Y = float(y_trunc) if y_trunc is not None else float(jnp.max(jnp.abs(y)))
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


def integrate_velocity_from_hilbert(y: Array, huy: Array) -> Array:
    """Cumulative trapezoid ``U(y)=∫_0^y HΩ`` with ``U(0)=0``.

    Twin of the torch ``_integrate_from_zero`` used by
    ``wholeline_hp_hu_from_omega``. This is **not** the ``OperatorBlock``
    ``integral`` role (activation antiderivative windows).
    """
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    huy = jnp.asarray(huy, dtype=jnp.float64).reshape(-1)
    order = jnp.argsort(y)
    y_s = y[order]
    f_s = huy[order]
    dy = jnp.diff(y_s)
    trap = 0.5 * (f_s[1:] + f_s[:-1]) * dy
    pref = jnp.concatenate([jnp.zeros((1,), dtype=huy.dtype), jnp.cumsum(trap)])
    i0 = int(jnp.argmin(jnp.abs(y_s)))
    u_s = pref - pref[i0]
    out = jnp.empty_like(huy)
    return out.at[order].set(u_s)


__all__ = [
    "hilbert_gl_panel",
    "hilbert_gl_single_raw_tail",
    "hilbert_gl_split_core",
    "hilbert_tail_power_mapped",
    "hilbert_wholeline_hp",
    "integrate_velocity_from_hilbert",
]
