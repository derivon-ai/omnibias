# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Randomized brute-force soundness sweep for the QUBO / Ising gap certificate.

Both lower-bound families are covered: the spectral / box-QP seal and the
Lasserre SOS bound. Every assertion is exact -- see
``omnibias-discrete/tests/test_gap_soundness_regression.py`` for why a certified
bound cannot be checked with a tolerance.

The front-ends (``max_cut``, ``max_independent_set``) are swept too: they build
their own penalty matrices, so a sign or scale slip there produces a certificate
that is internally coherent and still wrong about the graph problem it claims
to solve.
"""

from __future__ import annotations

import numpy as np
import pytest
from _enclosure import assert_lower_bound
from omnibias.qubo import (
    QUBOProblem,
    brute_force_min,
    certify_qubo_gap,
    decode_qubo,
    max_cut,
    max_independent_set,
)


def _random_qubo(seed: int, *, n_max: int = 6) -> QUBOProblem:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, n_max + 1))
    m = rng.standard_normal((n, n))
    return QUBOProblem(m + m.T, rng.standard_normal(n), const=0.4)


def _random_graph(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = (rng.random((n, n)) < 0.45).astype(float)
    a = np.triu(a, 1)
    return a + a.T


def test_spectral_lower_bound_never_exceeds_the_true_minimum() -> None:
    for seed in range(120):
        problem = _random_qubo(seed)
        _, e_min = brute_force_min(problem)
        assignment, _ = decode_qubo(problem, seed=seed)
        cert = certify_qubo_gap(problem, assignment, kind="spectral")
        assert_lower_bound(cert.lower_bound, e_min, what=f"QUBO minimum (seed={seed})")


@pytest.mark.slow
def test_sos_lower_bound_never_exceeds_the_true_minimum() -> None:
    pytest.importorskip("omnibias.sos")
    for seed in range(40):
        problem = _random_qubo(seed, n_max=5)
        _, e_min = brute_force_min(problem)
        assignment, _ = decode_qubo(problem, seed=seed)
        cert = certify_qubo_gap(
            problem, assignment, kind="sos", level=1, bisection_steps=20
        )
        assert_lower_bound(cert.lower_bound, e_min, what=f"QUBO minimum (seed={seed})")


def test_sos_lower_bound_smoke() -> None:
    """A cheap always-on slice of the slow sweep above."""
    pytest.importorskip("omnibias.sos")
    for seed in range(5):
        problem = _random_qubo(seed, n_max=4)
        _, e_min = brute_force_min(problem)
        assignment, _ = decode_qubo(problem, seed=seed)
        cert = certify_qubo_gap(
            problem, assignment, kind="sos", level=1, bisection_steps=12
        )
        assert_lower_bound(cert.lower_bound, e_min, what=f"QUBO minimum (seed={seed})")


@pytest.mark.parametrize("front_end", ["max_cut", "max_independent_set"])
def test_front_end_certificates_bound_their_own_brute_force(front_end: str) -> None:
    """A front-end's penalty encoding must not break the sandwich it reports."""
    build = {"max_cut": max_cut, "max_independent_set": max_independent_set}[front_end]
    for seed in range(60):
        adjacency = _random_graph(seed, n=6)
        problem = build(adjacency)
        _, e_min = brute_force_min(problem)
        assignment, _ = decode_qubo(problem, seed=seed)
        cert = certify_qubo_gap(problem, assignment, kind="spectral")
        assert_lower_bound(
            cert.lower_bound, e_min, what=f"{front_end} minimum (seed={seed})"
        )
        ulp = 64.0 * np.finfo(float).eps * max(abs(e_min), 1.0)
        assert cert.energy >= e_min - ulp
