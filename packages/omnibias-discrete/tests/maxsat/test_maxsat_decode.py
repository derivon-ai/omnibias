# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""MaxSAT decoding: an upper bound that finds a satisfying assignment when one exists."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from omnibias.discrete import brute_force_min, decode
from omnibias.discrete.maxsat import max_sat


def test_decode_is_an_upper_bound() -> None:
    prob = max_sat([[1, -2], [2, 3], [-1, -3], [1, 2, 3]])
    _, e_dec = decode(prob, n_starts=16)
    _, e_min = brute_force_min(prob)
    assert e_dec >= e_min - 1e-9


def test_satisfiable_instance_decodes_to_zero_violation() -> None:
    prob = max_sat([[1, -2], [2, 3], [-1, -3]])  # satisfiable
    assignment, energy = decode(prob, n_starts=16)
    _, e_min = brute_force_min(prob)
    assert e_min == pytest.approx(0.0)
    assert energy == pytest.approx(0.0)
    assert float(prob.energy(np.array(assignment, dtype=float))) == pytest.approx(0.0)


def test_unsatisfiable_core_has_positive_minimum() -> None:
    prob = max_sat([[1], [-1]])  # x0 and ~x0 cannot both hold
    _, e_min = brute_force_min(prob)
    assert e_min == pytest.approx(1.0)


def test_brute_force_matches_exhaustive_scan() -> None:
    prob = max_sat([[1, -2], [2, 3], [-1, -3]], weights=[1.0, 2.0, 0.5])
    _, e_best = brute_force_min(prob)
    rows = np.array(list(itertools.product([0.0, 1.0], repeat=prob.n)))
    assert e_best == pytest.approx(float(np.min(np.asarray(prob.energy(rows)))))
