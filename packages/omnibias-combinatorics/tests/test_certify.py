# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The LP-dual certificate: soundness, the tight sandwich, and graceful degradation."""

from __future__ import annotations

import builtins

import numpy as np
import pytest
from omnibias.combinatorics import (
    AssignmentProblem,
    GraphicMatroid,
    MatroidProblem,
    MinCostFlowProblem,
    TransportProblem,
    UniformMatroid,
    brute_force_min,
    certify_gap,
    classical_optimum,
    decode,
    max_flow_value,
)

K = 8


def _assignment(seed: int) -> AssignmentProblem:
    return AssignmentProblem(np.random.default_rng(seed).random((6, 6)))


def _matroid(seed: int) -> MatroidProblem:
    return MatroidProblem(np.random.default_rng(seed).standard_normal(8), UniformMatroid(8, 3))


@pytest.mark.parametrize("seed", range(K))
def test_sandwich_holds_against_brute_force(seed: int) -> None:
    """lower_bound <= brute_force_min <= decoded objective, and the bound is interval-sealed."""
    for prob in (_assignment(seed), _matroid(seed)):
        _, bf = brute_force_min(prob)
        x, decoded = decode(prob)
        cert = certify_gap(prob, x)
        assert cert.lower_bound <= bf + 1e-7
        assert bf <= decoded + 1e-7
        assert cert.certified is True  # convex installed -> Neumaier-Shcherbina sealed
        assert cert.is_sound
        assert cert.absolute_gap >= -1e-9


@pytest.mark.parametrize("seed", range(K))
def test_gap_is_tight_because_polytope_is_integral(seed: int) -> None:
    """Decoding the optimum gives a certified relative gap ~ 0 (integral polytope)."""
    for prob in (_assignment(seed), _matroid(seed)):
        x, _ = classical_optimum(prob)
        cert = certify_gap(prob, x)
        assert cert.relative_gap < 1e-6


@pytest.mark.parametrize("seed", range(4))
def test_transport_and_flow_sandwich(seed: int) -> None:
    rng = np.random.default_rng(seed)
    supply = np.array([2.0, 3.0, 1.0])
    demand = np.array([1.0, 2.0, 2.0, 1.0])
    tprob = TransportProblem(rng.random((3, 4)), supply, demand)
    xt, ot = classical_optimum(tprob)
    ct = certify_gap(tprob, xt)
    assert ct.lower_bound <= ot + 1e-7 and ct.is_sound and ct.certified

    arcs = ((0, 1), (0, 2), (1, 3), (2, 3), (1, 2))
    cap = rng.integers(1, 5, size=len(arcs)).astype(float)
    cost = rng.random(len(arcs))
    mf = max_flow_value(4, arcs, cap, 0, 3)
    fprob = MinCostFlowProblem(4, arcs, cost, cap, 0, 3, value=mf)
    xf, of = classical_optimum(fprob)
    cf = certify_gap(fprob, xf)
    assert cf.lower_bound <= of + 1e-7 and cf.is_sound and cf.certified


def test_graphic_matroid_certificate() -> None:
    mat = GraphicMatroid(4, ((0, 1), (1, 2), (2, 0), (2, 3)))
    prob = MatroidProblem(np.array([0.9, 0.8, 0.7, 0.6]), mat)
    x, _ = classical_optimum(prob)
    _, bf = brute_force_min(prob)
    cert = certify_gap(prob, x)
    assert cert.lower_bound <= bf + 1e-7
    assert cert.polytope == "matroid"
    assert cert.relative_gap < 1e-6


def test_solution_length_validated() -> None:
    with pytest.raises(ValueError, match="entries"):
        certify_gap(_assignment(0), np.zeros(5))


def test_graceful_degrade_without_convex(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without omnibias-convex the bound is the valid float LP value (certified=False)."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "omnibias.convex" or name.startswith("omnibias.convex."):
            raise ImportError("omnibias-convex disabled for this test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    prob = _assignment(3)
    x, _ = classical_optimum(prob)
    cert = certify_gap(prob, x)
    assert cert.certified is False
    assert cert.method == "lp_float"
    _, bf = brute_force_min(prob)
    assert cert.lower_bound <= bf + 1e-7  # still a valid lower bound
