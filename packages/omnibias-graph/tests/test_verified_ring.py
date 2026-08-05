# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rigorous ring-graph eigenvalue certificate (verified interval enclosure).

The cycle graph ``C_n`` has the analytic combinatorial-Laplacian eigenpairs

.. math::

    \lambda_k = 2 - 2\cos(2\pi k / n), \qquad
    v_k(i) = \cos(2\pi k i / n),

since ``(A v)_i = v_{i-1} + v_{i+1} = 2\cos(2\pi k/n) v_i`` and ``L = 2I - A``.

We feed the *analytic* eigenpair to
:func:`omnibias.core.verified.eig.symmetric_eigenvalue_residual_enclosure`, whose
symmetric-residual theorem returns an outward-rounded interval **guaranteed to
contain a true eigenvalue** of ``L`` within ``rho = ||Lv - theta v|| / ||v||`` of
the Rayleigh quotient. A tiny ``rho`` certifies -- not samples -- that the
analytic ring eigenvalue is a genuine eigenvalue. This reuses the same verified
substrate that backs the omnibias spectral-gap certificates.
"""

from __future__ import annotations

import math

import pytest
from omnibias.core.verified.eig import (
    rayleigh_quotient,
    symmetric_eigenvalue_residual_enclosure,
)


def _ring_laplacian(n: int) -> list[list[float]]:
    lap = [[0.0] * n for _ in range(n)]
    for i in range(n):
        lap[i][i] = 2.0
        lap[i][(i + 1) % n] = -1.0
        lap[i][(i - 1) % n] = -1.0
    return lap


def _cos_mode(n: int, k: int) -> list[float]:
    return [math.cos(2.0 * math.pi * k * i / n) for i in range(n)]


@pytest.mark.parametrize("n", [6, 8, 12])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_ring_eigenvalue_is_certified(n: int, k: int) -> None:
    lap = _ring_laplacian(n)
    v = _cos_mode(n, k)
    lam = 2.0 - 2.0 * math.cos(2.0 * math.pi * k / n)

    enclosure = symmetric_eigenvalue_residual_enclosure(lap, v)
    # The certified interval must bracket the analytic eigenvalue ...
    assert enclosure.lo <= lam <= enclosure.hi
    # ... and the enclosure of a *true* eigenpair is razor-thin (rho ~ 1e-15).
    assert (enclosure.hi - enclosure.lo) < 1e-9


@pytest.mark.parametrize("n", [6, 10])
def test_rayleigh_quotient_matches_analytic(n: int) -> None:
    lap = _ring_laplacian(n)
    for k in range(1, n // 2):
        v = _cos_mode(n, k)
        lam = 2.0 - 2.0 * math.cos(2.0 * math.pi * k / n)
        rq = rayleigh_quotient(lap, v)
        assert rq.lo <= lam <= rq.hi


def test_fiedler_gap_is_certified_positive() -> None:
    # The spectral gap lambda_1 > 0 of a connected ring is certified: the
    # smallest non-trivial mode encloses a strictly positive eigenvalue.
    n = 8
    lap = _ring_laplacian(n)
    v = _cos_mode(n, 1)
    enclosure = symmetric_eigenvalue_residual_enclosure(lap, v)
    assert enclosure.lo > 0.0
