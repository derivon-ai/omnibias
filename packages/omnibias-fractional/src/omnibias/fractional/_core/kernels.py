# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic fractional-derivative kernels (numpy).

These build the weight arrays for the grid-based fractional operators. They are
pure numpy (float64), so the torch and jax ops that consume them agree to
floating-point tolerance.

.. warning::

   These are the **grid-based** kernels. Fractional derivatives here are
   *non-local numerical approximations on a grid* whose accuracy is set by the
   grid resolution -- **not** the exact closed-form sigma-tower derivatives that
   the rest of omnibias provides; see ``FRACTIONAL_DERIVATIONS.md`` for the error
   budget. (The package's *closed-form* analytic operator lives in
   ``omnibias.fractional.torch.ops.analytic`` / ``...jax.ops.analytic`` and does
   not use these kernels.)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def gl_weights(alpha: float, n: int) -> NDArray[np.float64]:
    r"""Grunwald-Letnikov weights ``w_k = (-1)^k \binom{\alpha}{k}``, ``k=0..n-1``.

    Built by the stable recurrence ``w_0 = 1``,
    ``w_k = w_{k-1} (1 - (alpha + 1) / k)``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    w = np.empty(n, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = w[k - 1] * (1.0 - (alpha + 1.0) / k)
    return w


def gl_matrix(alpha: float, n: int, h: float) -> NDArray[np.float64]:
    r"""Lower-triangular Grunwald-Letnikov operator matrix of shape ``(n, n)``.

    Row ``i`` realises ``D^alpha f[i] = h^{-alpha} sum_{k=0}^{i} w_k f[i-k]``.
    """
    if h <= 0.0:
        raise ValueError(f"grid spacing h must be > 0, got {h}")
    w = gl_weights(alpha, n)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        mat[i, : i + 1] = w[i::-1]
    return mat * (h ** (-alpha))


def spectral_multiplier(
    alpha: float, n: int, length: float,
) -> NDArray[np.complex128]:
    r"""Fourier multiplier ``(i k)^alpha`` for the spectral fractional derivative.

    ``k`` are the angular wavenumbers for a periodic domain of period ``length``
    sampled at ``n`` points. The zero mode is set to zero.
    """
    if length <= 0.0:
        raise ValueError(f"length must be > 0, got {length}")
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    mult = (1j * k) ** alpha
    mult[0] = 0.0
    return mult.astype(np.complex128)


__all__ = ["gl_matrix", "gl_weights", "spectral_multiplier"]
