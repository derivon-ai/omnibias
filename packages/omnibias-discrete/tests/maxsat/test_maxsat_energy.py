# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""MaxSAT energy: manual checks, batch consistency, and the faithful polynomial encoding."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from omnibias.discrete.maxsat import max_sat


def test_energy_counts_violated_weight() -> None:
    # (x0 or ~x1) and (x1 or x2): clause 1 violated iff x0=0, x1=1; clause 2 iff x1=0, x2=0.
    prob = max_sat([[1, -2], [2, 3]])
    assert prob.energy(np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0)  # only clause 1
    assert prob.energy(np.array([0.0, 0.0, 0.0])) == pytest.approx(1.0)  # only clause 2
    assert prob.energy(np.array([1.0, 0.0, 1.0])) == pytest.approx(0.0)  # both satisfied


def test_weights_scale_the_violation() -> None:
    prob = max_sat([[1], [-1]], weights=[2.0, 3.0])  # (x0) w=2, (~x0) w=3
    assert prob.energy(np.array([0.0])) == pytest.approx(2.0)
    assert prob.energy(np.array([1.0])) == pytest.approx(3.0)


def test_batch_energy_matches_pointwise() -> None:
    prob = max_sat([[1, -2], [2, 3], [-1, -3]])
    rows = np.array(list(itertools.product([0.0, 1.0], repeat=prob.n)))
    batch = np.asarray(prob.energy(rows))
    pointwise = np.array([float(prob.energy(r)) for r in rows])
    assert np.allclose(batch, pointwise)


def test_polynomial_reproduces_energy_on_every_vertex() -> None:
    pytest.importorskip("omnibias.sos")
    prob = max_sat([[1, -2], [2, 3], [-1, -3]], weights=[1.5, 2.0, 0.5])
    poly = prob.to_polynomial()
    for bits in itertools.product([0, 1], repeat=prob.n):
        x = np.array(bits, dtype=float)
        assert float(prob.energy(x)) == pytest.approx(float(poly.evaluate(list(bits))))
