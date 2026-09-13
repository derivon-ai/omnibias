# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Independent checks of the regular-parabola estimates; not a Hilbert proof."""

from __future__ import annotations

import cmath
import math
import random
from fractions import Fraction

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.quadratic_graphic import (
    certify_outer_schwarzian,
    normalized_outer_error,
    normalized_outer_limit_jet,
    outer_first_variation_ray,
    outer_splitting_asymptotics,
    regular_outer_multiplier,
)

mp = pytest.importorskip("mpmath")


def test_exact_parabola_coordinate_and_invariant_identity() -> None:
    # Rational arithmetic checks the identities independently of interval code.
    rng = random.Random(16)
    for _ in range(80):
        x = Fraction(rng.randint(-20, 20), 7)
        c = Fraction(rng.randint(8, 30), 7)
        w = Fraction(rng.randint(-3, 3), 100)
        y = w + (x*x - c) / 2
        dx, dy = x - y + x*x, c*x + x*x + x*y
        q = ((x + 1)**2 + c - 1) / 2
        assert dx == q - w
        assert dy - x*dx == 2*x*w
        if q - w == 0 or q + w == 0:
            continue
        wx = 2*x*w / (q - w)
        jx = wx / (q + w)**2 - 2*w*((x + 1) + wx) / (q + w)**3
        assert jx == -2*w / (q + w)**3


@pytest.mark.parametrize("c", [1.25, 2.0, 5.0])
def test_regular_multiplier_contains_independent_quadrature(c: float) -> None:
    rng = random.Random(19)
    sections = [(-2.0, 2.0), (-6.0, 3.0), (0.0, 1.0)]
    sections += [(-rng.uniform(0, 4), rng.uniform(0, 4)) for _ in range(6)]
    with mp.workdps(60):
        for left, right in sections:
            actual = mp.exp(mp.quad(lambda x: 4*x / (x*x + 2*x + c), [left, right]))
            bound = regular_outer_multiplier(c, left, right)
            assert mp.mpf(bound.lo) <= actual <= mp.mpf(bound.hi)


def test_limit_derivatives_contain_grid_and_random_high_precision_values() -> None:
    rng = random.Random(23)
    points = [i / 512 for i in range(-16, 17)]
    points += [rng.uniform(-1/32, 1/32) for _ in range(20)]
    with mp.workdps(60):
        for c_parameter in (1.5, 2.0, 4.0):
            c = mp.exp(-4*mp.pi / mp.sqrt(c_parameter - 1))

            def limit(s, multiplier=c):
                target = multiplier*s / (1 + s)**2
                return 2*target / (1 - 2*target + mp.sqrt(1 - 4*target))

            for point in points:
                enclosure = normalized_outer_limit_jet(c_parameter, point)
                for order, bound in enumerate(enclosure):
                    actual = mp.diff(limit, mp.mpf(point), order)
                    assert mp.mpf(bound.lo) <= actual <= mp.mpf(bound.hi)


def test_parameter_and_section_boxes_enclose_all_sampled_derivatives() -> None:
    c_box, z_box = Interval(2.0, 2.01), Interval(-1/128, 1/128)
    enclosure = normalized_outer_limit_jet(c_box, z_box)
    rng = random.Random(29)
    with mp.workdps(60):
        for _ in range(30):
            c_parameter = rng.uniform(c_box.lo, c_box.hi)
            point = rng.uniform(z_box.lo, z_box.hi)
            c = mp.exp(-4*mp.pi / mp.sqrt(c_parameter - 1))

            def limit(s, multiplier=c):
                target = multiplier*s / (1 + s)**2
                return 2*target / (1 - 2*target + mp.sqrt(1 - 4*target))

            for order, bound in enumerate(enclosure):
                actual = mp.diff(limit, mp.mpf(point), order)
                assert mp.mpf(bound.lo) <= actual <= mp.mpf(bound.hi)


def _normalized_passage(c: float, radius: float, initial: complex, steps: int) -> complex:
    # Independent direct RK4 in the rational x-time equation; not a verifier.
    def rhs(x: float, z: complex) -> complex:
        q = ((x + 1)**2 + c - 1) / 2
        return z*((x - 1) + (x + 1)*z) / (q*(1 - z))

    h = 2*radius / steps
    z = initial
    for index in range(steps):
        x = -radius + index*h
        k1 = rhs(x, z)
        k2 = rhs(x + h/2, z + h*k1/2)
        k3 = rhs(x + h/2, z + h*k2/2)
        k4 = rhs(x + h, z + h*k3)
        z += h*(k1 + 2*k2 + 2*k3 + k4)/6
    return z


@pytest.mark.parametrize("radius", [6.0, 12.0, 24.0])
def test_complex_disk_comparison_against_direct_flow(radius: float) -> None:
    c_parameter = 2.0
    c = math.exp(-4*math.pi)
    comparison = normalized_outer_error(c_parameter, radius)
    points = [cmath.rect(1/16, index*math.pi/4) for index in range(8)]
    rng = random.Random(31)
    points += [cmath.rect(rng.uniform(0, 1/16), rng.uniform(-math.pi, math.pi)) for _ in range(8)]
    for point in points:
        target = c*point / (1 + point)**2
        limit = 2*target / (1 - 2*target + cmath.sqrt(1 - 4*target))
        direct = _normalized_passage(c_parameter, radius, point, 4000)
        refined = _normalized_passage(c_parameter, radius, point, 8000)
        assert abs(refined - direct) < comparison.uniform_value_error / 1000
        assert abs(refined - limit) < comparison.uniform_value_error


def test_schwarzian_certificate_covers_parameter_interval_and_unbounded_cutoffs() -> None:
    c_box = Interval(2.0, Interval.from_rational(Fraction(201, 100)).hi)
    result = certify_outer_schwarzian(c_box, 1e9)
    assert result.certified_negative
    assert result.certified_for_larger_cutoffs
    assert result.cells == 32
    assert not result.unresolved_cells
    assert all(bound is not None and bound.hi < 0 for bound in result.schwarzian_bounds)
    later = normalized_outer_error(c_box, 2e9)
    assert later.uniform_value_error < result.comparison.uniform_value_error


def test_small_cutoff_remains_unresolved() -> None:
    result = certify_outer_schwarzian(2.0, 6.0)
    assert not result.certified_negative
    assert not result.certified_for_larger_cutoffs
    assert len(result.unresolved_cells) == result.cells


@pytest.mark.parametrize("c", [1.0, 0.0, Interval(0.9, 2.0), math.inf])
def test_singular_or_unbounded_parameter_is_refused(c) -> None:
    with pytest.raises(ValueError):
        normalized_outer_error(c, 10.0)


@pytest.mark.parametrize("radius", [5.9, -1.0, math.inf])
def test_invalid_cutoff_is_refused(radius: float) -> None:
    with pytest.raises(ValueError):
        normalized_outer_error(2.0, radius)


def test_invalid_section_and_subdivision_are_refused() -> None:
    with pytest.raises(ValueError):
        normalized_outer_limit_jet(2.0, Interval(-0.1, 0.1))
    with pytest.raises(ValueError):
        certify_outer_schwarzian(2.0, 10.0, cells=0)


@pytest.mark.parametrize("c_parameter", [1.5, 2.0, 5.0])
def test_splitting_coefficients_against_independent_full_line_integrals(c_parameter: float) -> None:
    result = outer_splitting_asymptotics(c_parameter)
    with mp.workdps(55):
        a = mp.sqrt(c_parameter - 1)

        def weight(x):
            q = (x*x + 2*x + c_parameter)/2
            return mp.exp(4/a*(mp.atan((x + 1)/a) - mp.pi/2)) / q**3

        sources = (
            lambda x: -x*x,
            lambda x: -x*x*(x*x - c_parameter)/2,
            lambda x: (c_parameter*c_parameter - x**4)/4,
        )
        for source, bound in zip(sources, (result.alpha, result.mu2, result.mu3), strict=True):
            actual = mp.quad(
                lambda x, source_fn=source: weight(x)*source_fn(x),
                [-mp.inf, -1, 0, 1, mp.inf],
            )/2
            assert mp.mpf(bound.lo) <= actual <= mp.mpf(bound.hi)


def test_first_variation_ray_contains_finite_section_integrals() -> None:
    c_box = Interval(2.0, Interval.from_rational(Fraction(201, 100)).hi)
    result = outer_first_variation_ray(c_box, 16.0)
    rng = random.Random(37)
    samples = [(2.0, 16.0), (2.01, 32.0), (2.005, 128.0)]
    samples += [(rng.uniform(2, 2.01), rng.uniform(16, 64)) for _ in range(5)]
    with mp.workdps(55):
        for c_parameter, radius in samples:
            a = mp.sqrt(c_parameter - 1)

            def q(x, c=c_parameter):
                return (x*x + 2*x + c)/2

            def weight(x, root=a):
                return mp.exp(4/root*(mp.atan((x + 1)/root) - mp.pi/2)) / q(x)**3

            prefactor = q(radius)/radius**2 * mp.exp(4/a*mp.atan(a/(radius + 1)))
            sources = (
                lambda x: -x*x,
                lambda x, c=c_parameter: -x*x*(x*x - c)/2,
                lambda x, c=c_parameter: (c*c - x**4)/4,
            )
            for source, bound in zip(sources, (result.alpha, result.mu2, result.mu3), strict=True):
                actual = prefactor*mp.quad(
                    lambda x, source_fn=source: weight(x)*source_fn(x),
                    [-radius, -1, 0, 1, radius],
                )
                assert mp.mpf(bound.lo) <= actual <= mp.mpf(bound.hi)


def test_large_cutoff_ray_has_negative_parameter_variations() -> None:
    result = outer_first_variation_ray(Interval(2.0, 2.01), 10000.0)
    assert result.alpha.hi < 0
    assert result.mu2.hi < 0
    assert result.mu3.hi < 0
    for c, radius in ((2.0, 5.0), (100.0, 6.0)):
        with pytest.raises(ValueError):
            outer_first_variation_ray(c, radius)
