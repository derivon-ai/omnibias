# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The generic certified optimality-gap sandwich and its graceful degradation."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest
from omnibias.discrete import brute_force_min, certify_gap, decode


def test_sos_sandwich_is_sound(make_toy) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("omnibias.sos")
    rng = np.random.default_rng(0)
    for seed in range(4):
        m = rng.standard_normal((3, 3))
        prob = make_toy(m + m.T, c=rng.standard_normal(3), const=0.3)
        _, e_min = brute_force_min(prob)
        assignment, _ = decode(prob, seed=seed, n_starts=16)
        cert = certify_gap(prob, assignment, level=1, bisection_steps=20)
        assert cert.method in ("sos", "negative_coeff")
        assert cert.lower_bound <= e_min + 1e-6  # lower bound never exceeds the true min
        assert cert.energy >= e_min - 1e-9  # decoded energy is an upper bound
        assert cert.is_sound and cert.absolute_gap >= -1e-9


def test_sos_bound_is_sealed(make_toy) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("omnibias.sos")
    prob = make_toy([[1.0, -1.0], [-1.0, 1.0]])  # E = (x0 - x1)^2
    assignment, _ = decode(prob)
    cert = certify_gap(prob, assignment, level=1, bisection_steps=24)
    assert cert.method == "sos"
    assert cert.certified is True and cert.sealed is not None and cert.level == 1


def test_degrades_gracefully_without_sos(make_toy, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setitem(sys.modules, "omnibias.sos", None)
    prob = make_toy([[0.0, -1.0], [-1.0, 0.0]], c=[0.5, 0.5])
    assignment, _ = decode(prob)
    cert = certify_gap(prob, assignment)
    assert cert.method == "none"  # degraded, not crashed
    assert math.isinf(cert.lower_bound) and cert.lower_bound < 0
    assert cert.certified is False and cert.is_sound


def test_rejects_a_non_binary_point(make_toy) -> None:  # type: ignore[no-untyped-def]
    prob = make_toy([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="binary"):
        certify_gap(prob, np.array([0.5, 0.5]))


def test_rejects_a_wrong_length_point(make_toy) -> None:  # type: ignore[no-untyped-def]
    prob = make_toy([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="length"):
        certify_gap(prob, np.array([0.0, 1.0, 0.0]))
