# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""TimeScale jump operators, graininess, point classification, and grids."""

from __future__ import annotations

import pytest
from omnibias.timescale import finite, h_integers, quantum, reals


def test_reals_is_dense_with_zero_graininess() -> None:
    R = reals()
    for t in (-2.0, 0.0, 1.5, 100.0):
        assert R.sigma(t) == t
        assert R.rho(t) == t
        assert R.mu(t) == 0.0
        assert R.is_right_dense(t)
        assert R.is_left_dense(t)
        assert R.contains(t)


def test_h_integers_uniform_graininess() -> None:
    H = h_integers(0.5)
    assert H.sigma(1.0) == 1.5
    assert H.rho(1.0) == 0.5
    assert H.mu(1.0) == 0.5
    assert H.nu(1.0) == 0.5
    assert H.is_right_scattered(1.0)
    assert H.is_left_scattered(1.0)
    assert H.grid(0.0, 2.0) == (0.0, 0.5, 1.0, 1.5, 2.0)
    assert H.contains(1.5)
    assert not H.contains(1.25)


def test_quantum_scale() -> None:
    Q = quantum(2.0)
    assert Q.sigma(4.0) == 8.0
    assert Q.rho(4.0) == 2.0
    assert Q.mu(4.0) == 4.0  # (q - 1) t
    assert Q.sigma(0.0) == 0.0  # 0 is right-dense
    assert Q.is_right_dense(0.0)
    assert Q.grid(1.0, 8.0) == (1.0, 2.0, 4.0, 8.0)
    assert Q.contains(1.0) and Q.contains(2.0) and Q.contains(0.0)
    assert not Q.contains(3.0)


def test_quantum_graininess_vanishes_as_q_to_one() -> None:
    t = 5.0
    mus = [quantum(q).mu(t) for q in (2.0, 1.5, 1.1, 1.01)]
    assert mus == sorted(mus, reverse=True)
    assert mus[-1] == pytest.approx(0.05)


def test_finite_scale() -> None:
    T = finite((0.0, 1.0, 3.0, 7.0))
    assert T.sigma(1.0) == 3.0
    assert T.rho(3.0) == 1.0
    assert T.mu(1.0) == 2.0
    assert T.sigma(7.0) == 7.0  # right-maximal
    assert T.is_right_dense(7.0)
    assert T.grid(0.0, 3.0) == (0.0, 1.0, 3.0)


def test_factory_validation() -> None:
    with pytest.raises(ValueError):
        h_integers(0.0)
    with pytest.raises(ValueError):
        quantum(1.0)  # needs q > 1
    with pytest.raises(ValueError):
        quantum(0.5)
    with pytest.raises(ValueError):
        finite((1.0,))  # needs >= 2 points
    with pytest.raises(ValueError):
        reals().grid(0.0, 1.0)  # continuum has no grid
