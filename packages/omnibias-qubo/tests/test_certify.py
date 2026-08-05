# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The certified optimality-gap sandwich and its graceful degradation."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from omnibias.qubo import (
    QUBOProblem,
    brute_force_min,
    certify_qubo_gap,
    decode_qubo,
)


def _random_qubo(n: int, seed: int) -> QUBOProblem:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n))
    return QUBOProblem(m + m.T, rng.standard_normal(n), const=0.4)


def test_spectral_sandwich_is_sound() -> None:
    for seed in range(6):
        prob = _random_qubo(5, seed)
        _, e_min = brute_force_min(prob)
        assignment, _ = decode_qubo(prob, seed=seed)
        cert = certify_qubo_gap(prob, assignment, kind="spectral")
        assert cert.method == "spectral"
        assert cert.lower_bound <= e_min + 1e-6  # lower bound never exceeds the true min
        assert cert.energy >= e_min - 1e-9  # decoded energy is an upper bound
        assert cert.is_sound
        assert cert.absolute_gap >= -1e-9


def test_spectral_is_interval_sealed_with_convex() -> None:
    pytest.importorskip("omnibias.convex")
    prob = _random_qubo(4, 3)
    assignment, _ = decode_qubo(prob)
    cert = certify_qubo_gap(prob, assignment, kind="spectral")
    assert cert.certified is True


def test_sos_bound_is_sound_certified_and_sealed() -> None:
    pytest.importorskip("omnibias.sos")
    prob = _random_qubo(3, 0)
    _, e_min = brute_force_min(prob)
    assignment, _ = decode_qubo(prob, n_starts=16)
    cert = certify_qubo_gap(prob, assignment, kind="sos", level=1, bisection_steps=20)
    assert cert.method == "sos"
    assert cert.certified is True
    assert cert.sealed is not None
    assert cert.level == 1
    assert cert.lower_bound <= e_min + 1e-6
    assert cert.is_sound


def test_sos_is_at_least_as_tight_as_spectral() -> None:
    pytest.importorskip("omnibias.sos")
    pytest.importorskip("omnibias.convex")
    prob = _random_qubo(4, 5)
    assignment, _ = decode_qubo(prob)
    sos = certify_qubo_gap(prob, assignment, kind="sos", level=1, bisection_steps=20)
    spectral = certify_qubo_gap(prob, assignment, kind="spectral")
    # Both are valid lower bounds; the SOS bound is expected to be tighter (larger).
    assert sos.lower_bound >= spectral.lower_bound - 1e-6


def test_sos_falls_back_to_spectral_without_sos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "omnibias.sos", None)
    prob = _random_qubo(4, 1)
    _, e_min = brute_force_min(prob)
    assignment, _ = decode_qubo(prob)
    cert = certify_qubo_gap(prob, assignment, kind="sos")
    assert cert.method == "spectral"  # degraded, not crashed
    assert cert.sealed is None
    assert cert.lower_bound <= e_min + 1e-6
    assert cert.is_sound


def test_spectral_degrades_to_float_without_convex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "omnibias.convex", None)
    prob = _random_qubo(4, 2)
    _, e_min = brute_force_min(prob)
    assignment, _ = decode_qubo(prob)
    cert = certify_qubo_gap(prob, assignment, kind="spectral")
    assert cert.certified is False  # not interval-sealed
    assert cert.lower_bound <= e_min + 1e-6  # but still a valid bound
    assert cert.is_sound


def test_certify_rejects_a_non_binary_point() -> None:
    prob = _random_qubo(3, 0)
    with pytest.raises(ValueError, match="binary"):
        certify_qubo_gap(prob, np.array([0.5, 0.5, 0.5]), kind="spectral")


def test_certify_rejects_unknown_kind() -> None:
    prob = _random_qubo(3, 0)
    assignment, _ = decode_qubo(prob)
    with pytest.raises(ValueError, match="unknown kind"):
        certify_qubo_gap(prob, assignment, kind="exact")
