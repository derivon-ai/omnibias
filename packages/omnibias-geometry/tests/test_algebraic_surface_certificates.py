# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact-source and critical-level checks for the eight-sphere quartic."""

from dataclasses import replace
from fractions import Fraction as Q
from itertools import product

import pytest
from omnibias.core.realization.polynomial import SparsePolynomial as P
from omnibias.geometry.algebraic_surfaces import (
    certify_separable_quartic_surface,
    replay_separable_quartic_surface,
    separable_quartic_polynomial,
)


@pytest.mark.parametrize("epsilon", [Q(1, 16), Q(1, 2), Q(999, 1000), Q(1, 10**50)])
def test_exact_eight_spheres_and_complex_smoothness(epsilon):
    polynomial = separable_quartic_polynomial(epsilon)
    cert = certify_separable_quartic_surface(polynomial)
    assert cert.epsilon == epsilon
    assert cert.affine_critical_levels == (0, 1, 2, 3)
    assert cert.critical_level_gap == min(epsilon, 1 - epsilon)
    assert cert.squared_radius_gap == 1 - epsilon > 0
    assert cert.component_count == 8 and cert.component_genera == (0,) * 8
    assert set(cert.component_orthants) == set(product((-1, 1), repeat=3))
    assert replay_separable_quartic_surface(polynomial, cert)


def test_coefficients_derived_independently_and_critical_points_exhausted():
    x, y, z, w = (P.variable(4, i) for i in range(4))
    epsilon = Q(2, 7)
    polynomial = x**4 + y**4 + z**4 - 2 * w**2 * (x**2 + y**2 + z**2) + (3 - epsilon) * w**4
    cert = certify_separable_quartic_surface(polynomial)
    for point in product((-1, 0, 1), repeat=3):
        homogeneous = (*point, 1)
        assert all(polynomial.derivative(i).evaluate(homogeneous) == 0 for i in range(3))
        assert polynomial.evaluate(homogeneous) == point.count(0) - epsilon != 0
    assert cert.component_count == 8
    # Every nonzero rational infinity point has a nonzero spatial partial.
    for point in product((-2, -1, 0, 1, 2), repeat=3):
        if any(point):
            assert any(polynomial.derivative(i).evaluate((*point, 0)) != 0 for i in range(3))


def test_surface_source_and_proof_operand_tampering_rejected():
    polynomial = separable_quartic_polynomial(Q(1, 16))
    cert = certify_separable_quartic_surface(polynomial)
    for forged in (
        replace(cert, epsilon=Q(1, 8)),
        replace(cert, affine_critical_levels=(Q(0),)),
        replace(cert, critical_level_gap=Q(1)),
        replace(cert, squared_radius_gap=Q(1)),
        replace(cert, component_orthants=cert.component_orthants[:-1]),
        replace(cert, component_genera=(1,) * 8),
        replace(cert, source_digest="forged"),
    ):
        assert not replay_separable_quartic_surface(polynomial, forged)
    assert not replay_separable_quartic_surface(separable_quartic_polynomial(Q(1, 8)), cert)
    x = P.variable(4, 0)
    with pytest.raises(ValueError, match="coefficients"):
        certify_separable_quartic_surface(polynomial + x**4)
    assert not replay_separable_quartic_surface(polynomial + x**4, cert)
    with pytest.raises(ValueError, match="coefficients"):
        certify_separable_quartic_surface(polynomial + x)
    with pytest.raises(ValueError, match="X,Y,Z,W"):
        certify_separable_quartic_surface(P.variable(3, 0)**4)


def test_independent_rational_real_point_in_every_certified_orthant():
    source = separable_quartic_polynomial(Q(9, 16))
    cert = certify_separable_quartic_surface(source)
    for a, b, c in cert.component_orthants:
        point = (a, b, Q(c, 2), 1)
        assert source.evaluate(point) == 0
        assert source.derivative(2).evaluate(point) != 0
        # The displayed point maps to (0,0,-3/4) on the u-sphere.
        assert tuple(x * x - 1 for x in point[:3]) == (0, 0, -Q(3, 4))


@pytest.mark.parametrize("epsilon", [0, 1, -1, 2, Q(101, 100)])
def test_outside_topology_range_rejected_from_actual_coefficients(epsilon):
    with pytest.raises(ValueError, match="0 < epsilon < 1"):
        separable_quartic_polynomial(epsilon)
    x, y, z, w = (P.variable(4, i) for i in range(4))
    source = (x**2 - w**2)**2 + (y**2 - w**2)**2 + (z**2 - w**2)**2 - epsilon * w**4
    with pytest.raises(ValueError, match="0 < epsilon < 1"):
        certify_separable_quartic_surface(source)


@pytest.mark.parametrize("epsilon", [0, 1])
def test_rejected_boundary_parameters_really_have_complex_singularities(epsilon):
    x, y, z, w = (P.variable(4, i) for i in range(4))
    source = (x**2 - w**2)**2 + (y**2 - w**2)**2 + (z**2 - w**2)**2 - epsilon * w**4
    singular = (1, 1, 1, 1) if epsilon == 0 else (0, 1, 1, 1)
    assert source.evaluate(singular) == 0
    assert all(source.derivative(i).evaluate(singular) == 0 for i in range(4))


def test_float_parameter_rejected():
    with pytest.raises(TypeError, match="exact"):
        separable_quartic_polynomial(0.25)
