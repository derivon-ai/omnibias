# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The Gilmore-Lawler bound + ``certify_gap(kind="glb")``: sound, scalable, non-tight.

GLB is the QAP-specific flagship bound: **sound** at every size (exact integer arithmetic for
integer instances, outward-rounded intervals for floats), far **tighter** than the generic
spectral / box-QP bound, and -- unlike the Lasserre / SOS SDP -- **scalable** to realistic
block counts. It is still NP-hard-honest: a lower bound with a generally non-zero gap, never
an exactness claim.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.nphard import (
    brute_force_min,
    certify_gap,
    gilmore_lawler_bound,
    placement_qap,
    qap,
    schedule,
)
from omnibias.nphard._core.qap import perm_to_x, qap_brute_force


def _random_qap(dim: int, seed: int, *, integer: bool = True) -> object:
    rng = np.random.default_rng(seed)
    flow = rng.integers(0, 9, size=(dim, dim)).astype(float)
    dist = rng.integers(0, 9, size=(dim, dim)).astype(float)
    flow = flow + flow.T
    dist = dist + dist.T
    if not integer:  # induce half-integers -> the outward-rounded interval path
        flow = flow / 2.0
        dist = dist / 2.0
    np.fill_diagonal(flow, 0.0)
    np.fill_diagonal(dist, 0.0)
    return qap(flow, dist)


def _placement(n: int, grid: tuple[int, int], seed: int) -> object:
    rng = np.random.default_rng(seed)
    m = rng.integers(0, 5, size=(n, n)).astype(float)
    m = m + m.T
    np.fill_diagonal(m, 0.0)
    return placement_qap(m, grid)


def test_glb_is_sound_on_every_tiny_seed_integer_and_float() -> None:
    """Hard gate: ``GLB <= brute_force_min`` on all tiny-N instances (both arithmetic paths)."""
    for integer in (True, False):
        for dim in (3, 4, 5):
            for seed in range(6):
                prob = _random_qap(dim, seed, integer=integer)
                lower, sound = gilmore_lawler_bound(prob)
                _, e_bf = brute_force_min(prob)
                assert sound
                assert lower <= e_bf + 1e-9, f"GLB {lower} exceeds optimum {e_bf} (dim={dim}, seed={seed})"


def test_glb_is_integer_valued_for_integer_instances() -> None:
    """Integer ``F``, ``D`` -> exact integer arithmetic -> an integral bound."""
    lower, sound = gilmore_lawler_bound(_random_qap(5, 1, integer=True))
    assert sound and lower == float(int(lower))


def test_glb_is_far_tighter_than_the_spectral_bound() -> None:
    """GLB dominates the generic box-QP / eigenvalue bound at placement sizes."""
    for seed in range(5):
        prob = _placement(6, (2, 3), seed)
        x_opt, _ = qap_brute_force(prob)
        cg = certify_gap(prob, x_opt, kind="glb")
        cs = certify_gap(prob, x_opt, kind="spectral", bisection_steps=16)
        assert cg.lower_bound > cs.lower_bound  # strictly tighter (spectral is ~vacuous here)


def test_certify_glb_returns_a_sound_gilmore_lawler_certificate() -> None:
    prob = _placement(6, (2, 3), 0)
    x_opt, e_opt = brute_force_min(prob)
    cert = certify_gap(prob, x_opt, kind="glb")
    assert cert.method == "gilmore_lawler"
    assert cert.level == 0
    assert cert.is_sound and cert.certified
    assert cert.sealed is None  # interval-sound, not SOS-sealed (mirrors the spectral path)
    assert cert.lower_bound <= e_opt + 1e-9 <= cert.energy + 1e-9  # a valid sandwich


def test_certify_glb_upper_is_the_pure_wirelength_objective() -> None:
    """The GLB certificate's upper bound is the penalty-free permutation objective."""
    prob = _placement(6, (2, 3), 1)
    x_opt, _ = brute_force_min(prob)
    cert = certify_gap(prob, x_opt, kind="glb")
    assert cert.energy == pytest.approx(float(prob.objective(np.asarray(x_opt, dtype=float))))


def test_certify_glb_on_a_non_qap_family_raises_typeerror() -> None:
    sp = schedule(np.array([3.0, 1.0, 2.0, 4.0]), 2)
    with pytest.raises(TypeError, match="QAP-specific"):
        certify_gap(sp, np.zeros(sp.n), kind="glb")


def test_certify_glb_on_a_non_permutation_point_raises_valueerror() -> None:
    prob = _placement(6, (2, 3), 0)
    with pytest.raises(ValueError, match="permutation"):
        certify_gap(prob, np.zeros(prob.n), kind="glb")


def test_glb_scales_where_brute_force_and_sos_are_infeasible() -> None:
    """GLB certifies a 12-module placement (brute force 12! and the 144-var SOS SDP die)."""
    prob = _placement(12, (3, 4), 2)
    x = perm_to_x(tuple(range(12)), 12)  # any valid placement as the certified upper point
    cert = certify_gap(prob, x, kind="glb")
    assert cert.is_sound and cert.certified
    assert cert.lower_bound <= cert.energy + 1e-9  # sound sandwich at a size no exact method reaches
    assert cert.relative_gap >= 0.0
