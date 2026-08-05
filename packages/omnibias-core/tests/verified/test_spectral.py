# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verified periodic spectral operators: Hilbert transform + trig design.

The Hilbert circulant is exercised two ways:

* **Rigorously** -- its entries enclose the exact closed-form kernel, it is
  structurally antisymmetric, it annihilates constants, and applying it to an
  exactly representable vector encloses the exact real matrix-vector product
  (computed independently in high precision).  These are theorem-grade.
* **Approximately** -- it reproduces ``H cos(kx) = sin(kx)`` on float-sampled
  modes to ``O(n * ulp)``.  This is a correctness sanity check, not a containment
  claim: sampling a mode at *float* arguments perturbs the input by ~1 ulp per
  node, so the rigorous output legitimately differs from ``math.sin`` at that
  level.
"""

from __future__ import annotations

import math

import mpmath as mp
import pytest
from omnibias.core.verified import (
    Interval,
    cos_matrix,
    cos_point,
    hilbert_circulant,
    matvec,
    sin_matrix,
    sin_point,
    uniform_nodes,
)
from omnibias.core.verified.spectral import _hilbert_kernel


def _exact_kernel(n: int, m: int) -> mp.mpf:
    with mp.workdps(60):
        if m % 2 == 0:
            return mp.mpf(0)
        a = mp.pi * m / n
        return mp.mpf(2) / n * mp.cos(a) / mp.sin(a)


def test_design_matrices_enclose_truth() -> None:
    nodes = uniform_nodes(8)
    ks = [1.0, 2.0, 3.0]
    cmat = cos_matrix(nodes, ks)
    smat = sin_matrix(nodes, ks)
    for j, x in enumerate(nodes):
        for i, k in enumerate(ks):
            assert cmat[j][i].contains(math.cos(k * x))
            assert smat[j][i].contains(math.sin(k * x))
            assert cmat[j][i].width < 1e-12
            assert smat[j][i].width < 1e-12


@pytest.mark.parametrize("n", [8, 16, 32])
def test_kernel_encloses_closed_form(n: int) -> None:
    kernel = _hilbert_kernel(n)
    for m in range(n):
        truth = float(_exact_kernel(n, m))
        assert kernel[m].contains(truth)
        if m % 2 == 0:
            assert kernel[m].lo == 0.0 and kernel[m].hi == 0.0


@pytest.mark.parametrize("n", [8, 16, 32])
def test_kernel_is_antisymmetric(n: int) -> None:
    kernel = _hilbert_kernel(n)
    for m in range(n):
        s = kernel[m] + kernel[(n - m) % n]
        assert s.contains(0.0)


@pytest.mark.parametrize("n", [8, 16])
def test_matvec_encloses_exact_product(n: int) -> None:
    """On an exactly representable input, the interval matvec encloses truth."""
    hmat = hilbert_circulant(n)
    v_int = [((7 * j + 3) % 11) - 5 for j in range(n)]  # small integers, exact
    v = [Interval.point(float(t)) for t in v_int]
    out = matvec(hmat, v)
    with mp.workdps(60):
        for j in range(n):
            exact = mp.mpf(0)
            for col in range(n):
                exact += _exact_kernel(n, (j - col) % n) * v_int[col]
            assert out[j].contains(float(exact))


def test_hilbert_kills_constant() -> None:
    n = 16
    hmat = hilbert_circulant(n)
    ones = [Interval.point(1.0) for _ in range(n)]
    out = matvec(hmat, ones)
    for v in out:
        assert v.contains(0.0)
        assert abs(v.lo) < 1e-12 and abs(v.hi) < 1e-12


@pytest.mark.parametrize("n", [8, 16, 32])
def test_hilbert_mode_action_approx(n: int) -> None:
    """Sanity: H cos(kx)~sin(kx), H sin(kx)~-cos(kx) to O(n*ulp)."""
    nodes = uniform_nodes(n)
    hmat = hilbert_circulant(n)
    tol = 50 * n * 2.3e-16
    for k in range(1, n // 2):
        hc = matvec(hmat, [cos_point(k * x) for x in nodes])
        hs = matvec(hmat, [sin_point(k * x) for x in nodes])
        for j, x in enumerate(nodes):
            assert abs(hc[j].mid - math.sin(k * x)) < tol
            assert abs(hs[j].mid + math.cos(k * x)) < tol


def test_hilbert_requires_even_grid() -> None:
    with pytest.raises(ValueError):
        hilbert_circulant(7)
