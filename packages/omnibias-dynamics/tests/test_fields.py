# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Convenience field builders return consistent (field, jacobian) pairs."""

from __future__ import annotations

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.ode import TaylorSeries
from omnibias.dynamics import (
    harmonic_oscillator,
    hopf_normal_form,
    linear_system,
    radial_logistic,
)


def _eval_field(field, point: list[float]) -> list[Interval]:
    series = [TaylorSeries([Interval.point(p)]) for p in point]
    return [s.coeffs[0] for s in field(series)]


def test_harmonic_field_is_rotation() -> None:
    f, j = harmonic_oscillator(2.0)
    val = _eval_field(f, [1.0, 0.0])
    assert val[0].contains(0.0) and val[1].contains(-2.0)
    jac = j([Interval.point(1.0), Interval.point(0.0)])
    assert jac[0][1].contains(2.0) and jac[1][0].contains(-2.0)


def test_linear_system_matches_matrix() -> None:
    a = [[0.5, -1.0], [2.0, -0.3]]
    f, j = linear_system(a)
    val = _eval_field(f, [1.0, 1.0])
    assert val[0].contains(0.5 - 1.0)
    assert val[1].contains(2.0 - 0.3)
    jac = j([Interval.point(0.0), Interval.point(0.0)])
    for i in range(2):
        for k in range(2):
            assert jac[i][k].contains(a[i][k])


def test_hopf_jacobian_trace_on_cycle() -> None:
    # On the unit cycle the divergence is tr DF = 2 - 4 r^2 = -2 at r = 1.
    _, j = hopf_normal_form(1.0)
    jac = j([Interval.point(1.0), Interval.point(0.0)])
    trace = jac[0][0] + jac[1][1]
    assert trace.contains(-2.0)


def test_radial_fixed_point_has_zero_velocity() -> None:
    f, _ = radial_logistic(1.0)
    val = _eval_field(f, [1.0])
    assert val[0].contains(0.0)
