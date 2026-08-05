# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Rigorous optimality-gap certificate: soundness, tightness hierarchy, honesty."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.routing import (
    RoutingProblem,
    certify_tour_gap,
    decode_tour,
    held_karp_dp,
)

KINDS = ("assignment", "flow", "held_karp")


def _problem(n: int, seed: int) -> RoutingProblem:
    return RoutingProblem.from_coords(np.random.default_rng(seed).random((n, 2)))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("kind", KINDS)
def test_lower_bound_is_sound(seed: int, kind: str) -> None:
    """The certified lower bound is provably <= the exact optimum (LP relaxation)."""
    prob = _problem(7, seed)
    _, opt = held_karp_dp(prob.cost)
    tour, _ = decode_tour(prob.cost)
    cert = certify_tour_gap(prob, tour, kind=kind)
    assert cert.lower_bound <= opt + 1e-7
    assert cert.certified is True  # convex is installed -> interval-sealed


@pytest.mark.parametrize("seed", range(6))
def test_sandwich_holds(seed: int) -> None:
    """lower_bound <= optimum <= tour_cost, and the reported gap is non-negative."""
    prob = _problem(8, seed)
    _, opt = held_karp_dp(prob.cost)
    tour, cost = decode_tour(prob.cost)
    cert = certify_tour_gap(prob, tour, kind="flow")
    assert cert.lower_bound <= opt + 1e-7
    assert opt <= cert.tour_cost + 1e-7
    assert cert.tour_cost == pytest.approx(cost, abs=1e-9)
    assert cert.absolute_gap >= -1e-9
    assert cert.relative_gap >= -1e-9
    assert cert.is_sound


@pytest.mark.parametrize("seed", range(4))
def test_relaxation_hierarchy(seed: int) -> None:
    """Tighter relaxations give higher (never invalid) lower bounds: assign<=flow<=HK<=opt."""
    prob = _problem(7, seed)
    _, opt = held_karp_dp(prob.cost)
    tour, _ = decode_tour(prob.cost)
    lb = {k: certify_tour_gap(prob, tour, kind=k).lower_bound for k in KINDS}
    assert lb["assignment"] <= lb["flow"] + 1e-6
    assert lb["flow"] <= lb["held_karp"] + 1e-6
    assert lb["held_karp"] <= opt + 1e-6


def test_better_tour_has_smaller_gap() -> None:
    """Certifying the exact-optimal tour yields a gap <= that of a worse tour."""
    prob = _problem(8, 2)
    opt_tour, _ = held_karp_dp(prob.cost)
    worse = (0, 1, 2, 3, 4, 5, 6, 7)  # a fixed, generally suboptimal tour
    g_opt = certify_tour_gap(prob, opt_tour, kind="flow").absolute_gap
    g_worse = certify_tour_gap(prob, worse, kind="flow").absolute_gap
    assert g_opt <= g_worse + 1e-9
    assert g_opt >= -1e-9


def test_held_karp_gap_is_small() -> None:
    """The Held-Karp bound is tight: its certified relative gap is modest on small n."""
    gaps = []
    for seed in range(5):
        prob = _problem(7, seed)
        tour, _ = decode_tour(prob.cost)
        gaps.append(certify_tour_gap(prob, tour, kind="held_karp").relative_gap)
    assert np.mean(gaps) < 0.10  # Held-Karp is near-exact for these instances


def test_invalid_tour_rejected() -> None:
    prob = _problem(6, 0)
    with pytest.raises(ValueError, match="permutation"):
        certify_tour_gap(prob, (0, 1, 2, 3, 4, 4), kind="flow")


def test_unknown_kind_rejected() -> None:
    prob = _problem(6, 0)
    tour, _ = decode_tour(prob.cost)
    with pytest.raises(ValueError, match="unknown relaxation kind"):
        certify_tour_gap(prob, tour, kind="nonsense")
