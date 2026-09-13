# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact chain rules and enclosure checks for the full quadratic unfolding."""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.quadratic_unfolding import (
    certify_unfolding_nonoscillation,
    family_chart_field,
    scaled_unfolding_field,
    turning_geometry,
    unfolding_parameters,
)


def _contains(bound: Interval, value: Fraction) -> None:
    assert Fraction(bound.lo) <= value <= Fraction(bound.hi)


def test_exact_chain_rule_from_original_field_and_positive_time() -> None:
    rng = random.Random(1601)
    for _ in range(100):
        nu = Fraction(rng.randint(1, 10), 1000)
        a, c = Fraction(rng.randint(1, 3), 2), Fraction(rng.randint(2, 6), 2)
        m1, m2, m3 = (Fraction(rng.randint(-4, 4), 4) for _ in range(3))
        xx, yy = Fraction(rng.randint(-20, 20), 7), Fraction(rng.randint(1, 20), 5)
        original_x, original_y = xx / nu, yy / nu**2
        fx = (
            a*original_x-original_y+original_x**2
            + nu*(m2+m3)*original_x*original_y+nu**2*m1*original_y**2
        )
        fy = c*original_x+original_x**2+original_x*original_y+nu*m3*original_y**2
        dx, dy = nu**2*fx, nu**3*fy  # tau=t/nu.
        p = unfolding_parameters(nu=nu, a=a, c=c, m1=m1, m2=m2, m3=m3)
        scaled = scaled_unfolding_field(xx, yy, p)
        _contains(scaled[0], dx)
        _contains(scaled[1], dy)
        v, z = xx / yy, 1 / yy
        dv, dz = z*(dx*yy-xx*dy)/yy**2, -z*dy/yy**2
        result = family_chart_field(v, z, p)
        _contains(result[0], dv)
        _contains(result[1], dz)
        assert 1 / nu > 0 and yy > 0  # d(tau)/dt and d(s)/d(tau).
        assert dv == m1+m2*v-nu*v**3-z+a*nu*v*z-c*nu**2*v**2*z
        assert dz == -z*(m3+v+nu*v**2+c*nu**2*v*z)


def test_boundary_extension_and_leading_energy_identity() -> None:
    for v, z, m1, m2, m3 in itertools.product(
        [Fraction(-1), Fraction(0), Fraction(1)], repeat=5
    ):
        if z <= 0:
            continue
        p = unfolding_parameters(nu=0, a=1, c=2, m1=m1, m2=m2, m3=m3)
        f, g = m1+m2*v-z, -z*(m3+v)
        actual = family_chart_field(v, z, p)
        _contains(actual[0], f)
        _contains(actual[1], g)
        u, d = v+m3, m1-m2*m3
        # E=u**2/2+d*log(z)-z, differentiated without evaluating a logarithm.
        assert u*f+(d/z-1)*g == m2*u**2


def test_turning_determinant_and_implicit_derivative_by_rational_chain_rule() -> None:
    rng = random.Random(1602)
    for _ in range(100):
        nu = Fraction(rng.randint(0, 10), 1000)
        a, c = Fraction(rng.randint(1, 3), 2), Fraction(rng.randint(2, 6), 2)
        v, z = Fraction(rng.randint(-9, 9), 10), Fraction(rng.randint(1, 39), 10)
        m1, m2 = Fraction(rng.randint(-4, 4), 4), Fraction(rng.randint(-4, 4), 4)
        m3 = -v-nu*v**2-c*nu**2*v*z  # Construct an exact point on N=0.
        assert abs(m3) < 1
        p = unfolding_parameters(nu=nu, a=a, c=c, m1=m1, m2=m2, m3=m3)
        ell = 1+2*nu*v+c*nu**2*z
        f_v = m2-3*nu*v**2+a*nu*z-2*c*nu**2*v*z
        f_z = -1+a*nu*v-c*nu**2*v**2
        g_v = -z*ell
        g_z = -(m3+v+nu*v**2)-2*c*nu**2*v*z
        jacobian_determinant = f_v*g_z-f_z*g_v
        graph_derivative = -c*nu**2*v/ell
        drift_derivative = f_z+f_v*graph_derivative
        d = -jacobian_determinant/z
        assert drift_derivative == -d/ell
        geometry = turning_geometry(v, z, p)
        _contains(geometry.normal, Fraction(0))
        _contains(geometry.normal_v, ell)
        _contains(geometry.determinant_factor, d)


def test_default_parameter_cover_has_strict_computed_margins() -> None:
    result = certify_unfolding_nonoscillation()
    assert result.certified_at_most_one_height_extremum
    assert not result.unresolved_obligations
    assert result.left_normal.hi < 0 and result.right_normal.lo > 0
    assert result.geometry.normal_v.lo > 0
    assert result.geometry.determinant_factor.lo > Fraction(9, 10)
    assert result.drift_slope is not None
    assert result.drift_slope.hi < Fraction(-4, 5)
    assert result.parameters.nu.lo == 0
    assert result.parameters.nu.hi >= Fraction(1, 100)
    assert result.parameters.m3.contains(0)


def test_grid_and_random_samples_lie_in_phase_parameter_enclosures() -> None:
    p = unfolding_parameters()
    check = certify_unfolding_nonoscillation(p)
    broad_field = family_chart_field(check.v_domain, check.z_domain, p)
    geometry = check.geometry
    samples = []
    # A deterministic phase grid at three different parameter corners, including nu=0.
    for v, z in itertools.product(
        [Fraction(k, 4) for k in range(-8, 9)],
        [Fraction(k, 4) for k in range(17)],
    ):
        for index in range(3):
            samples.append((v, z, Fraction(index, 200), Fraction(index+1, 2),
                            Fraction(index+1), Fraction(index-1), Fraction(1-index),
                            Fraction(index-1)))
    rng = random.Random(1603)
    for _ in range(200):
        samples.append((Fraction(rng.randint(-200, 200), 100),
                        Fraction(rng.randint(0, 400), 100),
                        Fraction(rng.randint(0, 100), 10000),
                        Fraction(rng.randint(50, 150), 100),
                        Fraction(rng.randint(100, 300), 100),
                        *(Fraction(rng.randint(-100, 100), 100) for _ in range(3))))
    for v, z, nu, a, c, m1, m2, m3 in samples:
        f = m1+m2*v-nu*v**3-z+a*nu*v*z-c*nu**2*v**2*z
        n = m3+v+nu*v**2+c*nu**2*v*z
        ell = 1+2*nu*v+c*nu**2*z
        k = 1-a*nu*v+c*nu**2*v**2
        f_v = m2-3*nu*v**2+a*nu*z-2*c*nu**2*v*z
        d = k*ell+c*nu**2*v*f_v
        _contains(broad_field[0], f)
        _contains(broad_field[1], -z*n)
        for enclosure, value in ((geometry.normal, n), (geometry.normal_v, ell),
                                 (geometry.k, k), (geometry.f_v, f_v),
                                 (geometry.determinant_factor, d)):
            _contains(enclosure, value)
        assert ell > 0 and d > 0


@pytest.mark.parametrize(
    "values,missing",
    [
        ({"nu": Interval(0, 1)}, "normal_strictly_increasing_in_v"),
        ({"m3": 3}, "normal_negative_on_left"),
        ({"m3": -3}, "normal_positive_on_right"),
        ({"nu": Fraction(-1, 100)}, "nu_nonnegative"),
        ({"a": 1000}, "positive_turning_determinant"),
    ],
)
def test_unknown_parameter_regions_remain_unresolved(values, missing) -> None:
    result = certify_unfolding_nonoscillation(unfolding_parameters(**values))
    assert not result.certified_at_most_one_height_extremum
    assert missing in result.unresolved_obligations


@pytest.mark.parametrize("name", ["nu", "a", "c", "m1", "m2", "m3"])
def test_nonfinite_parameters_are_refused(name: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        unfolding_parameters(**{name: math.inf})
