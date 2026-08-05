# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exactness of the QUBO <-> Ising maps and the SOS polynomial encoders."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.qubo import (
    IsingProblem,
    QUBOProblem,
    ising_to_qubo,
    qubo_to_ising,
)


def _random_qubo(n: int, seed: int) -> QUBOProblem:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n))
    return QUBOProblem(m + m.T, rng.standard_normal(n), const=float(rng.standard_normal()))


def test_qubo_to_ising_energy_matches_on_every_vertex() -> None:
    rng = np.random.default_rng(0)
    for seed in range(15):
        n = int(rng.integers(1, 7))
        prob = _random_qubo(n, seed)
        ising = qubo_to_ising(prob)
        for _ in range(30):
            x = rng.integers(0, 2, size=n).astype(float)
            s = 2.0 * x - 1.0  # the spin change of variables s = 2x - 1
            assert abs(float(prob.energy(x)) - float(ising.energy(s))) < 1e-9


def test_qubo_ising_qubo_roundtrip() -> None:
    rng = np.random.default_rng(1)
    n = 5
    prob = _random_qubo(n, 7)
    back = ising_to_qubo(qubo_to_ising(prob))
    for _ in range(50):
        x = rng.integers(0, 2, size=n).astype(float)
        assert abs(float(prob.energy(x)) - float(back.energy(x))) < 1e-9


def test_ising_to_qubo_energy_matches() -> None:
    rng = np.random.default_rng(2)
    n = 5
    m = rng.standard_normal((n, n))
    ising = IsingProblem(m + m.T, rng.standard_normal(n), const=0.3)
    qubo = ising_to_qubo(ising)
    for _ in range(50):
        s = rng.choice([-1.0, 1.0], size=n)
        x = (s + 1.0) / 2.0
        assert abs(float(ising.energy(s)) - float(qubo.energy(x))) < 1e-9


def test_ising_diagonal_is_folded_into_const() -> None:
    j = np.array([[3.0, 1.0], [1.0, -2.0]])  # nonzero diagonal
    ising = IsingProblem(j, np.zeros(2), const=0.0)
    assert np.allclose(np.diag(ising.J), 0.0)
    # s^T J s with the raw diagonal equals the stored (zero-diagonal) energy.
    for s in ([1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]):
        sv = np.array(s)
        raw = float(sv @ j @ sv)
        assert abs(float(ising.energy(sv)) - raw) < 1e-9


def test_to_polynomial_matches_energy_as_a_function() -> None:
    pytest.importorskip("omnibias.sos")
    from omnibias.qubo import to_polynomial

    rng = np.random.default_rng(3)
    n = 4
    prob = _random_qubo(n, 4)
    poly = to_polynomial(prob)
    for _ in range(30):
        x = rng.standard_normal(n)  # arbitrary real point: polynomial == energy everywhere
        assert abs(poly.evaluate(list(x)) - float(prob.energy(x))) < 1e-9


def test_boolean_constraints_pin_the_cube() -> None:
    pytest.importorskip("omnibias.sos")
    from omnibias.qubo import boolean_constraints

    n = 3
    cons = boolean_constraints(n)
    assert len(cons) == 2 * n
    # At every binary point the ideal generators vanish (both inequalities are tight).
    for bits in range(1 << n):
        x = [(bits >> k) & 1 for k in range(n)]
        for g in cons:
            assert abs(g.evaluate([float(v) for v in x])) < 1e-12
