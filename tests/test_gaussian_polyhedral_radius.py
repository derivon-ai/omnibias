# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite path-radius certificates versus original Gaussian roots and KKT."""

from fractions import Fraction as F

import pytest
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral_path import compile_gaussian_polyhedral_path
from omnibias.core.verified.gaussian_polyhedral_radius import (
    certify_gaussian_polyhedral_path_radius,
)
from test_gaussian_polyhedral_path import _bind, _mp, _scalar

PARAMETER_RADIUS = F(1, 20)
SPATIAL_RADIUS = F(1, 4)


def _cubic(active, order):
    path = compile_gaussian_polyhedral_path(_scalar(active), ((-1,),)+((0,),)*(order-1),
                                          ((0, 2),)+((0, 0),)*(order-1))
    assert path is not None
    return path


def _certify(path, radius):
    return certify_gaussian_polyhedral_path_radius(path, parameter_radius=PARAMETER_RADIUS,
                                                  spatial_radius=SPATIAL_RADIUS, radius=radius)


def _assert_contains(mp, interval, expected):
    assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)


@pytest.mark.parametrize("order,radius", [(3, F(1, 100000)), (7, F(1, 125))])
@pytest.mark.parametrize("active", [(), (0,)])
def test_cubic_face_point_and_signed_kkt_bound_hold_on_grid_including_radius_endpoint(order, radius, active):
    mp = pytest.importorskip("mpmath")
    path = _cubic(active, order)
    certificate = _certify(path, radius)
    assert certificate is not None and certificate.radius == radius
    assert certificate.path is path
    assert certificate.status == path.status == ("invalid" if active else "valid")
    assert certificate.eventual_active_indices == (None if active else ())
    assert certificate.analytic.series is path.series
    assert certificate.analytic.parameter_radius == PARAMETER_RADIUS
    assert certificate.analytic.spatial_radius == SPATIAL_RADIUS
    assert len(certificate.sign_bounds) == 1
    bound = certificate.sign_bounds[0]
    assert bound.sign == path.sign_tests[0] and bound.certified
    assert bound.sign.leading_order == 3
    assert bound.sign.leading_coefficient == (F(-1, 3) if active else F(2, 3))
    assert bound.margin is not None and bound.margin > 0
    expected_change = (PARAMETER_RADIUS+PARAMETER_RADIUS/(1-PARAMETER_RADIUS**2)) if active else SPATIAL_RADIUS
    assert bound.change_bound == expected_change
    polynomial_remainder = sum(abs(value)*radius**(n-3) for n, value in enumerate(bound.sign.coefficients) if n > 3)
    ratio = radius/PARAMETER_RADIUS
    scaled_tail = expected_change*ratio**(order+1)/((1-ratio)*radius**3)
    assert bound.polynomial_remainder_bound == polynomial_remainder
    assert bound.scaled_tail_bound == scaled_tail
    assert bound.margin == abs(bound.sign.leading_coefficient)-polynomial_remainder-scaled_tail

    with mp.workdps(140):
        for parameter in (F(0), radius/8, radius/4, radius/2, radius):
            t = _mp(mp, parameter)
            optimizer = mp.mpf(0) if not parameter else mp.findroot(
                lambda x, t=t: x-mp.tanh(x/2+t)+t, (-t**3, mp.mpf(0)),
            )
            face_point = mp.mpf(0) if active else optimizer
            enclosure = certificate.enclose(parameter)
            assert len(enclosure) == 1
            _assert_contains(mp, enclosure[0], face_point)
            point_polynomial = sum(_mp(mp, row[0])*t**n for n, row in enumerate(path.coefficients))
            assert abs(face_point-point_polynomial) <= _mp(mp, certificate.tail_bound(parameter))
            assert certificate.tail_bound(parameter) == certificate.analytic.tail_bound(parameter)
            weights = (mp.exp(mp.mpf(1)/4-(face_point+1)**2/4),
                       mp.exp(mp.mpf(1)/4+2*t-(face_point-1)**2/4))
            original_mean = (weights[1]-weights[0])/sum(weights)
            original_velocity = face_point-original_mean
            multiplier = -t-original_velocity if active else mp.mpf(0)
            assert abs(original_velocity+t+multiplier) < mp.mpf("1e-120")
            quantity = multiplier if active else -face_point
            polynomial = sum(_mp(mp, coefficient)*t**n for n, coefficient in enumerate(bound.sign.coefficients))
            if parameter:
                q = parameter/PARAMETER_RADIUS
                true_tail_bound = bound.change_bound*q**(order+1)/(1-q)
                assert abs(quantity-polynomial) <= _mp(mp, true_tail_bound)
                signed_scaled = (-quantity if active else quantity)/t**3
                assert signed_scaled >= _mp(mp, bound.margin)
                assert optimizer < 0 and (multiplier < 0 if active else quantity > 0)
            else:
                assert quantity == polynomial == 0
        if active:
            # A certified invalid face point is not certified as the
            # optimizer. At this endpoint the true optimizer even lies
            # outside the returned narrow face-point enclosure.
            assert optimizer < _mp(mp, certificate.enclose(radius)[0].lo)


def test_fully_constrained_face_has_zero_projected_diameter_but_nonzero_bias_multiplier_change():
    certificate = _certify(_cubic((0,), 7), F(1, 125))
    assert certificate is not None and certificate.status == "invalid"
    assert certificate.path.equilibrium.face.geometry.restriction.model.diameter_squared == 0
    assert certificate.analytic.diameter_bound == 0
    assert certificate.analytic.target_growth_bound == 0
    assert certificate.analytic.bias_growth_bound == 2*PARAMETER_RADIUS
    assert certificate.analytic.phase_half_width_bound == PARAMETER_RADIUS
    assert certificate.analytic.forcing_bound == 0
    assert certificate.sign_bounds[0].change_bound > PARAMETER_RADIUS
    assert certificate.path.coefficients == ((0,),)*8
    assert certificate.path.inequality_multiplier_coefficients[3] == (F(-1, 3),)


def test_larger_finite_radius_requires_more_orders_even_when_eventual_path_status_is_known():
    low, high = _cubic((), 3), _cubic((), 7)
    assert low.status == high.status == "valid"
    assert _certify(low, F(1, 125)) is None
    assert _certify(low, F(1, 100000)) is not None
    assert _certify(high, F(1, 125)) is not None
    assert _certify(_cubic((), 2), F(1, 100000)) is None
    assert certify_gaussian_polyhedral_path_radius(high, parameter_radius=1, spatial_radius=SPATIAL_RADIUS,
                                                   radius=F(1, 125)) is None


@pytest.mark.parametrize("active,expected_status", [((0,), "invalid"), ((1,), "valid")])
def test_redundant_singleton_keeps_invalid_witness_scope_and_structural_zero_bounds(active, expected_status):
    source = GaussianHeatMap(((0,),), (0,))
    equilibrium = _bind(source, ((1,), (-1,)), (0, 0), active, (0,), (0,))
    path = compile_gaussian_polyhedral_path(equilibrium, ((-1,), (F(-1, 2),), (0,)), ((7,), (0,), (-3,)))
    assert path is not None and path.status == expected_status
    certificate = _certify(path, F(1, 125))
    assert certificate is not None and certificate.status == expected_status
    assert certificate.eventual_active_indices == ((0, 1) if expected_status == "valid" else None)
    omitted = next(bound for bound in certificate.sign_bounds if bound.sign.quantity == "slack")
    selected = next(bound for bound in certificate.sign_bounds if bound.sign.quantity == "multiplier")
    assert omitted.sign.verdict == "zero" and omitted.sign.identity is not None and omitted.certified
    assert omitted.change_bound == omitted.polynomial_remainder_bound == omitted.scaled_tail_bound == omitted.margin == 0
    assert selected.certified and selected.margin is not None and selected.margin > 0
    for parameter in (F(0), F(1, 250), F(1, 125)):
        assert certificate.enclose(parameter)[0].contains(0)
        # Both inequalities impose x=0, irrespective of the selected
        # multiplier representation's eventual sign or certification.
        assert path.series.evaluate(parameter) == (0,)


def test_one_certified_negative_quantity_suffices_despite_an_unresolved_other_sign():
    source = GaussianHeatMap(((0, 0),), (0,))
    equilibrium = _bind(source, ((1, 0), (0, 1)), (0, 0), (0,), (0, 0), (0, 0))
    path = compile_gaussian_polyhedral_path(equilibrium, ((-1, 0), (0, 0), (0, 0)), ((0,), (0,), (0,)))
    assert path is not None and path.status == "invalid"
    certificate = _certify(path, F(1, 125))
    assert certificate is not None and certificate.status == "invalid"
    negative = next(bound for bound in certificate.sign_bounds if bound.sign.quantity == "multiplier")
    unknown = next(bound for bound in certificate.sign_bounds if bound.sign.quantity == "slack")
    assert negative.sign.verdict == "negative" and negative.certified and negative.margin > 0
    assert unknown.sign.verdict == "inconclusive" and not unknown.certified
    assert unknown.polynomial_remainder_bound is unknown.scaled_tail_bound is unknown.margin is None
    assert unknown.change_bound == SPATIAL_RADIUS
    assert certificate.eventual_active_indices is None
    assert all(interval.contains(0) for interval in certificate.enclose(F(1, 125)))
