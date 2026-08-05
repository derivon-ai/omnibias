# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Validated variational / monodromy flow: soundness + known Floquet oracles."""

from __future__ import annotations

import math
import random

import pytest
from _dynamics_helpers import harmonic_float, hopf_float, rk4
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.linalg import to_interval_matrix
from omnibias.core.verified.lohner import interval_matrix_exp
from omnibias.dynamics import (
    harmonic_oscillator,
    hopf_normal_form,
    linear_system,
    monodromy_determinant,
    monodromy_matrix,
    monodromy_trace,
    radial_logistic,
    spectral_radius_bound,
    variational_flow,
)
from omnibias.dynamics._core.variational import VariationalState

TWO_PI = 2.0 * math.pi


def test_harmonic_monodromy_is_identity() -> None:
    # The undamped oscillator rotates by 2*pi over a period: M(2*pi) = I.
    f, j = harmonic_oscillator(1.0)
    m = monodromy_matrix(f, j, [1.0, 0.0], TWO_PI, n_steps=400)
    assert m[0][0].contains(1.0) and m[1][1].contains(1.0)
    assert m[0][1].contains(0.0) and m[1][0].contains(0.0)
    assert monodromy_trace(m).contains(2.0)
    assert monodromy_determinant(m).contains(1.0)


def test_linear_fundamental_contains_matrix_exponential() -> None:
    # For y' = A y the fundamental matrix is exp(A t); the validated flow must
    # contain the (independently enclosed) true value diag(e^{-1}, e^{-2}).
    a = [[-1.0, 0.0], [0.0, -2.0]]
    f, j = linear_system(a)
    vf = variational_flow(f, j, [1.0, 1.0], 1.0 / 200, 200)
    m = vf.fundamental
    assert m[0][0].contains(math.exp(-1.0))
    assert m[1][1].contains(math.exp(-2.0))
    assert m[0][1].contains(0.0) and m[1][0].contains(0.0)


def test_linear_fundamental_agrees_with_direct_exp() -> None:
    a = [[0.0, 1.0], [-1.0, 0.0]]  # rotation generator
    f, j = linear_system(a)
    t = 0.7
    vf = variational_flow(f, j, [0.3, -0.4], t / 150, 150)
    scaled = [[Interval.point(a[i][k]) * Interval.point(t) for k in range(2)] for i in range(2)]
    direct = interval_matrix_exp(to_interval_matrix(scaled))
    # Both rigorously enclose exp(A t); their intersection must be non-empty.
    for i in range(2):
        for k in range(2):
            lo = max(vf.fundamental[i][k].lo, direct[i][k].lo)
            hi = min(vf.fundamental[i][k].hi, direct[i][k].hi)
            assert lo <= hi


def test_hopf_radial_floquet_multiplier() -> None:
    # Scalar radial reduction r' = r - r^3: monodromy over 2*pi at r=1 is the
    # non-trivial Floquet multiplier exp(-4*pi) = exp(integral of (1 - 3 r^2)).
    f, j = radial_logistic(1.0)
    m = monodromy_matrix(f, j, [1.0], TWO_PI, n_steps=600)
    assert m[0][0].contains(math.exp(-4.0 * math.pi))


def test_hopf_planar_determinant_is_liouville() -> None:
    # The 2-D limit cycle has multipliers {1, exp(-4*pi)}: det = exp(-4*pi),
    # trace = 1 + exp(-4*pi) (Liouville's formula on the cycle).
    f, j = hopf_normal_form(1.0)
    m = monodromy_matrix(f, j, [1.0, 0.0], TWO_PI, n_steps=800)
    mult = math.exp(-4.0 * math.pi)
    assert monodromy_determinant(m).contains(mult)
    assert monodromy_trace(m).contains(1.0 + mult)


def test_spectral_radius_bracket_is_sound() -> None:
    f, j = harmonic_oscillator(1.0)
    m = monodromy_matrix(f, j, [1.0, 0.0], TWO_PI, n_steps=400)
    rho = spectral_radius_bound(m)
    assert rho.lo <= 1.0 <= rho.hi  # rotation: every multiplier has modulus 1


def test_state_encloses_float_trajectory_harmonic() -> None:
    f, j = harmonic_oscillator(1.0)
    vf = variational_flow(f, j, [1.0, 0.0], 1.3 / 200, 200)
    truth = rk4(harmonic_float(1.0), [1.0, 0.0], 0.0, 1.3, 20000)
    box = vf.box()
    assert box[0].contains(truth[0]) and box[1].contains(truth[1])


def test_state_encloses_float_trajectory_hopf() -> None:
    f, j = hopf_normal_form(1.0)
    y0 = [0.5, 0.2]
    vf = variational_flow(f, j, y0, 2.0 / 400, 400)
    truth = rk4(hopf_float(1.0), y0, 0.0, 2.0, 40000)
    box = vf.box()
    assert box[0].contains(truth[0]) and box[1].contains(truth[1])


def test_flow_encloses_grid_and_random_initial_conditions() -> None:
    """Founding soundness sweep: over a dense grid AND a random sample of initial
    conditions, the validated Hopf flow box encloses the fine RK4 trajectory.

    The single-IC tests above check one trajectory; this sweeps the initial-
    condition space so a wrapping/rounding regression that breaks containment for
    some interior state fails CI. The RK4 oracle is 60x finer than the flow steps.
    """
    f, j = hopf_normal_form(1.0)
    ffloat = hopf_float(1.0)
    t1, n_flow, n_rk4 = 0.8, 60, 4000
    axis = [-0.35, 0.0, 0.35]
    ics: list[list[float]] = [[a, b] for a in axis for b in axis]
    rng = random.Random(11)
    ics.extend([rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35)] for _ in range(3))
    for y0 in ics:
        box = variational_flow(f, j, y0, t1 / n_flow, n_flow).box()
        truth = rk4(ffloat, y0, 0.0, t1, n_rk4)
        assert box[0].contains(truth[0]) and box[1].contains(truth[1])


def test_variational_state_initial_is_identity() -> None:
    vs = VariationalState.initial([0.0, 0.0, 0.0])
    assert vs.time == 0.0
    for i in range(3):
        for k in range(3):
            assert vs.fundamental[i][k].contains(1.0 if i == k else 0.0)


def test_invalid_arguments_raise() -> None:
    f, j = harmonic_oscillator(1.0)
    with pytest.raises(ValueError):
        variational_flow(f, j, [1.0, 0.0], 0.1, 0)
    with pytest.raises(ValueError):
        monodromy_matrix(f, j, [1.0, 0.0], -1.0)
