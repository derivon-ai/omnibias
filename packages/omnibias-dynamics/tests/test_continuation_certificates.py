# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Uniform rather than sampled continuation and explicit event obligations."""

import pytest
from omnibias.core.verified.interval import Interval as I
from omnibias.core.verified.transcend import cosh_iv, sinh_iv
from omnibias.dynamics.continuation import certify_event, certify_join, certify_segment


def test_uniform_nonlinear_segment_and_join():
    # Generic callbacks, exact positive square-root branch.
    def f(x, s):
        return [x[0] * x[0] - s]

    def j(x, s):
        return [[2 * x[0]]]

    left = certify_segment(f, j, I(1.0, 1.01), [1.0025], slope=[0.5], radius=0.007)
    right = certify_segment(f, j, I(1.01, 1.02), [1.0075], slope=[0.5], radius=0.007)
    assert left.certified and right.certified
    assert certify_join(f, j, left, right)
    for k in range(101):
        s = 1 + k / 10000
        assert left.enclosure[0].lo <= s**0.5 <= left.enclosure[0].hi
    assert not certify_segment(f, j, I(-1.0, -0.9), [0.0], radius=0.01).certified


def test_fold_augmented_root_with_nonzero_normal_form():
    def f(z):
        return [z[0] * z[0] - z[1], 2 * z[0] * z[2], (z[2] * z[2] - 1) / 2]

    def j(z):
        return [
            [2 * z[0], I.point(-1), I.point(0)],
            [2 * z[2], I.point(0), 2 * z[0]],
            [I.point(0), I.point(0), z[2]],
        ]

    def conditions(box):
        return {
            "transversality": I.point(-1),
            "quadratic": box[2] * box[2],
            "complement_gap": I.point(1),
        }

    event = certify_event("fold", f, j, [0.0, 0.0, 1.0], conditions)
    assert event.certified

    def bad(box):
        return {**conditions(box), "quadratic": I(-1, 1)}

    assert not certify_event("fold", f, j, [0.0, 0.0, 1.0], bad).certified
    with pytest.raises(ValueError, match="conditions"):
        certify_event("fold", f, j, [0.0, 0.0, 1.0], lambda box: {})


def test_hopf_equilibrium_trace_system_conditions():
    # For planar Hopf f=(p*x-y-x*r², x+p*y-y*r²), use equilibrium + trace=0.
    def f(z):
        x, y, p = z
        r = x * x + y * y
        return [p * x - y - x * r, x + p * y - y * r, 2 * p - 4 * r]

    def j(z):
        x, y, p = z
        return [
            [p - 3 * x * x - y * y, -1 - 2 * x * y, x],
            [1 - 2 * x * y, p - x * x - 3 * y * y, y],
            [-8 * x, -8 * y, I.point(2)],
        ]

    # The analytic normal form at the unique root has frequency/crossing 1, l1=-2.
    def bounds(z):
        # These are normal-form quantities at the only possible equilibrium
        # in this box: x=y=0 (dot f with (-y,x) gives x²+y²=0).
        return {
            "transversality": I.point(1),
            "lyapunov": I.point(-2),
            "frequency": I.point(1),
            "complement_gap": I.point(1),
            "resonance_gap": I.point(1),
        }

    assert certify_event("hopf", f, j, [0.0, 0.0, 0.0], bounds).certified


def test_bratu_analytic_family_fold_enclosure():
    def value(z):
        a, p = z
        return [p * cosh_iv(a / 2) ** 2 - 2 * a * a, p * sinh_iv(a) / 2 - 4 * a]

    def jacobian(z):
        a, p = z
        return [
            [p * sinh_iv(a) / 2 - 4 * a, cosh_iv(a / 2) ** 2],
            [p * cosh_iv(a) / 2 - 4, sinh_iv(a) / 2],
        ]

    def bounds(z):
        a, p = z
        return {
            "transversality": cosh_iv(a / 2) ** 2,
            "quadratic": (p * cosh_iv(a) / 2 - 4) / 2,
            "complement_gap": I.point(1),
        }

    event = certify_event(
        "fold", value, jacobian, [2.3993572805, 3.5138307191], bounds, radius=1e-6
    )
    assert event.certified and event.root is not None
    assert event.root.enclosure[0][0] < 2.399357280515 < event.root.enclosure[0][1]
    assert event.root.enclosure[1][0] < 3.513830719125 < event.root.enclosure[1][1]
