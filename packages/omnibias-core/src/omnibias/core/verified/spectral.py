# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified periodic spectral operators -- the nonlocal ingredient.

Boundary / Euler-type blow-up models (De Gregorio, Hou-Luo, Boussinesq) are
driven by a **nonlocal** operator: the Hilbert transform / Biot-Savart law.  This
module supplies it rigorously.

* :func:`hilbert_circulant` builds the discrete periodic Hilbert transform on ``N``
  equispaced nodes as a *verified* circulant :class:`Interval` matrix.  Its entries
  come from the spectral multiplier ``-i sgn(k)`` summed in high precision and
  rounded outward, so applying it to an interval vector is theorem-grade.  On the
  grid it reproduces ``H cos(kx) = sin(kx)``, ``H sin(kx) = -cos(kx)`` for the
  resolved modes ``1 <= k < N/2``.
* :func:`cos_matrix` / :func:`sin_matrix` build verified trig design matrices
  ``C[j,i] = cos(k_i x_j)`` / ``S[j,i] = sin(k_i x_j)`` for collocation in a
  Fourier basis (the basis in which the periodic Hilbert transform acts exactly).

The kernel sums use ``mpmath`` when available (the high-precision backend named in
the certified-evidence contract) and fall back to ulp-inflated libm otherwise.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence

from omnibias.core.verified.interval import Interval, _pred, _succ
from omnibias.core.verified.transcend import (
    MPMATH_DPS,
    cos_point,
    sin_point,
    strict_backend,
)


def cos_matrix(x_nodes: Sequence[float], k_values: Sequence[float]) -> list[list[Interval]]:
    """Verified ``C[j,i] = cos(k_i * x_j)``."""
    return [[cos_point(float(k) * float(x)) for k in k_values] for x in x_nodes]


def sin_matrix(x_nodes: Sequence[float], k_values: Sequence[float]) -> list[list[Interval]]:
    """Verified ``S[j,i] = sin(k_i * x_j)``."""
    return [[sin_point(float(k) * float(x)) for k in k_values] for x in x_nodes]


def _hilbert_kernel(n: int) -> list[Interval]:
    r"""Verified circulant row ``s_m`` of the discrete periodic Hilbert transform.

    ``(H f)_j = sum_l s_{(j-l) mod n} f_l``.  The Nyquist mode is dropped, and the
    finite sum ``(2/n) sum_{a=1}^{n/2-1} sin(2 pi a m / n)`` collapses to the exact
    closed form

        ``s_m = 0``                  (m even),
        ``s_m = (2/n) cot(pi m / n)`` (m odd),

    which is structurally antisymmetric (``s_{n-m} = -s_m``).  Even entries are set
    to *exactly* zero -- evaluating the cancelling sum numerically would otherwise
    leave a spurious non-zero residual.  Odd entries use mpmath interval arithmetic
    so the enclosure is rigorous through the irrational argument ``pi m / n``.
    """
    if n < 2 or n % 2 != 0:
        raise ValueError("periodic Hilbert grid size must be even and >= 2")
    try:
        mp = importlib.import_module("mpmath")
    except ImportError:  # pragma: no cover - environment dependent
        mp = None
    kernel: list[Interval] = []
    if mp is not None:
        prev = mp.iv.dps
        mp.iv.dps = MPMATH_DPS
        try:
            two_over_n = mp.iv.mpf(2) / mp.iv.mpf(n)
            for m in range(n):
                if m % 2 == 0:
                    kernel.append(Interval(0.0, 0.0))
                    continue
                arg = mp.iv.pi * m / n
                val = two_over_n * mp.iv.cos(arg) / mp.iv.sin(arg)
                kernel.append(Interval(_pred(float(val.a)), _succ(float(val.b))))
        finally:
            mp.iv.dps = prev
    else:  # pragma: no cover - exercised only without mpmath
        if strict_backend():
            raise RuntimeError(
                "transcend strict mode is on but mpmath is unavailable: refusing "
                "the conditionally-rigorous libm Hilbert-kernel fallback"
            )
        for m in range(n):
            if m % 2 == 0:
                kernel.append(Interval(0.0, 0.0))
                continue
            arg = math.pi * m / n
            v = 2.0 * math.cos(arg) / math.sin(arg) / n
            lo, hi = v, v
            for _ in range(32):
                lo = _pred(lo)
                hi = _succ(hi)
            kernel.append(Interval(lo, hi))
    return kernel


def hilbert_circulant(n: int) -> list[list[Interval]]:
    """Verified ``n x n`` discrete periodic Hilbert-transform matrix."""
    kernel = _hilbert_kernel(n)
    return [[kernel[(j - col) % n] for col in range(n)] for j in range(n)]


def uniform_nodes(n: int, period: float = 2.0 * math.pi) -> list[float]:
    """``n`` equispaced nodes on ``[0, period)``."""
    return [period * j / n for j in range(n)]


__all__ = [
    "cos_matrix",
    "hilbert_circulant",
    "sin_matrix",
    "uniform_nodes",
]
