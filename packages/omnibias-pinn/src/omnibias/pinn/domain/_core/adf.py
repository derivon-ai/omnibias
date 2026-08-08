# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Normalized approximate distance functions (pure numpy).

A raw R-function or algebraic implicit equation ``omega`` vanishes on the
boundary but generally has ``|grad omega| != 1``. The normalized ADF

.. math::

    \phi = \frac{\omega}{\sqrt{\omega^2 + |\nabla\omega|^2}}

satisfies ``phi = 0`` on the boundary and ``|grad phi| = 1`` to first order
(Sukumar & Srivastava, *CMAME* 2022), which is what the multiplicative hard-BC
ansatz ``u = g + phi * NN`` needs for well-conditioned higher derivatives.

Honesty: the gradient used here is a central finite difference of ``omega``
(numerical). Analytic gradients of the primitives in
:mod:`omnibias.pinn.domain._core.sdf` are available for the common cases and
preferred when constructing a torch distance callable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from omnibias.pinn.domain._core.sdf import SDF, evaluate_sdf

FloatArray = np.ndarray
GradFn = Callable[[FloatArray], FloatArray]


def normalize_adf(
    omega: FloatArray,
    grad_omega: FloatArray,
    *,
    eps: float = 1e-30,
) -> FloatArray:
    """Normalize ``omega`` by ``sqrt(omega^2 + |grad omega|^2)``."""
    omega = np.asarray(omega, dtype=float).reshape(-1)
    g = np.asarray(grad_omega, dtype=float)
    if g.ndim == 1:
        g = g.reshape(-1, 1)
    if g.shape[0] != omega.shape[0]:
        raise ValueError(
            f"grad_omega leading dim {g.shape[0]} != omega length {omega.shape[0]}"
        )
    denom = np.sqrt(omega * omega + np.sum(g * g, axis=-1) + float(eps))
    return cast(FloatArray, omega / denom)


def fd_gradient(
    sdf: SDF,
    X: FloatArray,
    *,
    h: float = 1e-6,
) -> FloatArray:
    """Central-difference gradient of ``sdf`` at ``X`` (numerical)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n, d = X.shape
    out = np.empty((n, d), dtype=float)
    for j in range(d):
        Xp = X.copy()
        Xm = X.copy()
        Xp[:, j] += h
        Xm[:, j] -= h
        out[:, j] = (evaluate_sdf(sdf, Xp) - evaluate_sdf(sdf, Xm)) / (2.0 * h)
    return out


def approximate_distance(
    sdf: SDF,
    X: FloatArray,
    *,
    grad_fn: GradFn | None = None,
    h: float = 1e-6,
    eps: float = 1e-30,
) -> FloatArray:
    """Normalized ADF ``phi`` of ``sdf`` at ``X``."""
    omega = evaluate_sdf(sdf, X)
    if grad_fn is None:
        grad = fd_gradient(sdf, X, h=h)
    else:
        grad = np.asarray(grad_fn(X), dtype=float)
    return normalize_adf(omega, grad, eps=eps)


__all__ = [
    "approximate_distance",
    "fd_gradient",
    "normalize_adf",
]
