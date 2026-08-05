# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""MaxSAT certification: the SOS gap sandwiches the brute-force optimum."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.discrete import brute_force_min, certify_gap, decode
from omnibias.discrete.maxsat import max_sat


def test_certified_gap_sandwiches_the_optimum() -> None:
    pytest.importorskip("omnibias.sos")
    prob = max_sat([[1, -2], [2, 3], [-1, -3]], weights=[1.5, 2.0, 0.5])
    _, e_min = brute_force_min(prob)
    assignment, _ = decode(prob, n_starts=16)
    cert = certify_gap(prob, assignment, level=2, bisection_steps=24)
    assert cert.lower_bound <= e_min + 1e-6  # rigorous lower bound
    assert cert.energy >= e_min - 1e-9  # decoded upper bound
    assert cert.is_sound


def test_certified_gap_on_an_unsatisfiable_core() -> None:
    pytest.importorskip("omnibias.sos")
    prob = max_sat([[1], [-1]])  # minimum violated weight is exactly 1
    _, e_min = brute_force_min(prob)
    assignment, energy = decode(prob)
    assert e_min == pytest.approx(1.0)
    cert = certify_gap(prob, assignment, level=1)
    assert cert.lower_bound <= e_min + 1e-6
    assert cert.energy == pytest.approx(e_min)  # decoder is exact on this tiny core
    assert cert.is_sound


def test_certify_rejects_a_non_binary_point() -> None:
    prob = max_sat([[1, -2], [2, 3]])
    with pytest.raises(ValueError, match="binary"):
        certify_gap(prob, np.array([0.5, 0.5, 0.5]))
