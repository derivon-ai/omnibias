# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent covariance and tangent-tail checks without optional backends."""

from fractions import Fraction as F
from importlib import import_module
from math import nextafter

import pytest
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified._gaussian_bias_response import (
    _bias_covariance,
    _interval_response,
    _projected_mean,
)
from omnibias.core.verified.interval import Interval


def _arithmetic(**kwargs):
    return _Arithmetic(RidgeBudget(**kwargs))


def _iv(value):
    return Interval.from_rational(F(value))


def _contains(interval, exact):
    assert F(interval.lo) <= exact <= F(interval.hi)


def _covariance(probabilities, centers, beta):
    # Independent pairwise identity, deliberately O(m²*d), with no helper
    # expectation or mean-times-mean covariance formula.
    return tuple(sum(probabilities[j]*probabilities[k]*(centers[j][axis]-centers[k][axis])*(beta[j]-beta[k])
                     for j in range(len(centers)) for k in range(j))
                 for axis in range(len(centers[0])))


@pytest.mark.parametrize("probabilities", [(F(1, 3),)*3, (F(1, 7), F(2, 7), F(4, 7)), (F(0), F(1), F(0))])
@pytest.mark.parametrize("beta", [(F(0), F(1), F(-2)), (F(2, 5), F(-4, 7), F(11, 3))])
def test_covariance_matches_independent_exact_pairwise_identity(probabilities, beta):
    centers = ((F(0), F(0)), (F(2, 3), F(-4, 5)), (F(-7, 2), F(3, 8)))
    result = _bias_covariance(tuple(_iv(p) for p in probabilities), centers, beta, _arithmetic())
    expected = _covariance(probabilities, centers, beta)
    for interval, value in zip(result, expected, strict=True):
        _contains(interval, value)


def test_interval_probability_box_contains_multiple_normalized_distributions():
    box = (Interval(.125, .5), Interval(.25, .625), Interval(.125, .5))
    centers = ((F(0), F(0)), (F(2), F(-1)), (F(-3), F(4)))
    beta = (F(-2), F(3), F(1))
    result = _bias_covariance(box, centers, beta, _arithmetic())
    for p in ((F(1, 8), F(5, 8), F(1, 4)), (F(1, 2), F(1, 4), F(1, 4)), (F(1, 4), F(1, 4), F(1, 2))):
        for interval, value in zip(result, _covariance(p, centers, beta), strict=True):
            _contains(interval, value)


def test_common_bias_gauge_cancels_exactly_before_binary_conversion():
    probabilities = (_iv(F(1, 3)), _iv(F(2, 3)))
    centers = ((F(0), F(0)), (F(1, 7), F(-3, 2)))
    beta = (F(1, 3), F(5, 7))
    shifted = tuple(value+10**1000 for value in beta)
    assert _bias_covariance(probabilities, centers, beta, _arithmetic()) == _bias_covariance(probabilities, centers, shifted, _arithmetic())


@pytest.mark.parametrize("beta", [(F(0), F(0)), (F(10**1000), F(10**1000))])
def test_constant_bias_is_exact_zero_even_for_wide_posterior_intervals(beta):
    result = _bias_covariance((Interval(0, 1),)*2, ((F(0), F(0)), (F(1), F(-2))), beta, _arithmetic())
    assert all(value.lo == value.hi == 0 for value in result)


def test_all_normal_projected_modes_give_zero_covariance_and_mean():
    centers = ((F(0), F(0)),)*3
    probabilities = (_iv(F(1, 3)),)*3
    covariance = _bias_covariance(probabilities, centers, (F(0), F(10**400), F(-10**400)), _arithmetic())
    mean = _projected_mean(probabilities, centers, _arithmetic())
    assert all(value.lo == value.hi == 0 for value in (*covariance, *mean))


def test_zero_projected_mode_can_affect_other_modes_through_normalization():
    centers = ((F(0), F(0)), (F(0), F(0)), (F(1), F(-1)))
    probabilities = (_iv(F(1, 4)), _iv(F(1, 2)), _iv(F(1, 4)))
    result = _bias_covariance(probabilities, centers, (F(0), F(1), F(0)), _arithmetic())
    assert result == (_iv(F(-1, 8)), _iv(F(1, 8)))


def test_per_axis_projected_zeros_are_preserved():
    centers = ((F(0), F(0), F(0)), (F(1), F(0), F(-2)), (F(-3), F(0), F(4)))
    probabilities = (Interval(.1, .3), Interval(.3, .6), Interval(.1, .6))
    result = _bias_covariance(probabilities, centers, (F(0), F(2), F(-3)), _arithmetic())
    mean = _projected_mean(probabilities, centers, _arithmetic())
    assert result[1].lo == result[1].hi == mean[1].lo == mean[1].hi == 0


def test_exact_products_can_cancel_extreme_units_before_float_conversion():
    # Neither the center nor its mean is finite binary64, but the requested
    # covariance is exactly 1/4 because the bias direction has inverse units.
    result = _bias_covariance((_iv(F(1, 2)),)*2, ((F(0),), (F(10**400),)),
                              (F(0), F(1, 10**400)), _arithmetic())
    assert result == (_iv(F(1, 4)),)


def test_underflow_interval_does_not_turn_a_positive_covariance_into_exact_zero():
    tiny = nextafter(0.0, 1.0)
    probabilities = (Interval(nextafter(1.0, 0.0), 1.0), Interval(0.0, tiny))
    result = _bias_covariance(probabilities, ((F(0),), (F(1),)), (F(0), F(1)), _arithmetic())
    p = F(1, 10**1000)
    _contains(result[0], p*(1-p))
    assert result[0].hi > 0


def test_projected_mean_clips_only_to_exact_normalized_weight_hull():
    centers = ((F(0), F(0)), (F(2), F(-3)), (F(5), F(-1)))
    result = _projected_mean((Interval(0, 1),)*3, centers, _arithmetic())
    assert result == (Interval(0, 5), Interval(-3, 0))


def test_projected_mean_avoids_converting_large_coefficients_before_small_weights():
    centers = ((F(0),), (F(10**400),))
    posterior = (Interval(nextafter(1.0, 0.0), 1.0), Interval(0, 1e-300))
    result = _projected_mean(posterior, centers, _arithmetic())
    _contains(result[0], F(10**400)*F(1, 10**300))
    assert 0 < result[0].hi < 1e101


def test_response_encloses_independent_geometric_inverse_with_tangent_tail():
    # R projects onto span(3,4); T=(1/3)R. Three Neumann terms have
    # Q=I+(1/3+1/9)R and exact remainder norm s*(1/3)^3/(1-1/3).
    r = ((F(9, 25), F(12, 25)), (F(12, 25), F(16, 25)))
    q = tuple(tuple(F(int(i == j))+F(4, 9)*r[i][j] for j in range(2)) for i in range(2))
    polynomial = tuple(tuple(_iv(v) for v in row) for row in q)
    forcing = (F(6), F(8))
    result = _interval_response(r, polynomial, F(2), F(1, 9), tuple(_iv(v) for v in forcing), _arithmetic())
    exact = (F(18), F(24))
    for interval, value in zip(result, exact, strict=True):
        _contains(interval, value)


def test_response_boxes_can_include_nontangent_points_without_false_normal_output():
    projector = ((F(0), F(0)), (F(0), F(1)))
    polynomial = ((_iv(1), _iv(0)), (_iv(0), _iv(2)))
    result = _interval_response(projector, polynomial, F(3), F(1, 2), (Interval(-100, 100), Interval(1, 2)), _arithmetic())
    assert result[0].lo == result[0].hi == 0
    # The true forcing has its first component zero; the coarser norm from
    # the enclosing box is safe, albeit intentionally not sharp.
    _contains(result[1], F(6))
    _contains(result[1], F(12))


def test_zero_forcing_skips_square_root_and_is_exact(monkeypatch):
    module = import_module("omnibias.core.verified._gaussian_bias_response")

    def forbidden(*args, **kwargs):
        pytest.fail("zero forcing evaluated a square root")

    monkeypatch.setattr(module, "_sqrt_upper", forbidden)
    result = _interval_response(((F(1),),), ((_iv(100),),), F(2), F(10), (_iv(0),), _arithmetic())
    assert result == (_iv(0),)


def test_exact_time_times_forcing_precedes_float_conversion():
    time = F(10**400)
    forcing = Interval.point(1e-300)
    result = _interval_response(((F(1),),), ((_iv(1),),), time, F(0), (forcing,), _arithmetic())
    _contains(result[0], time*F(forcing.lo))
    assert result[0].hi < 1e101


def test_component_tail_uses_projector_diagonal_and_outward_norm():
    # Empty polynomial (zero enclosure) isolates the remainder. For this
    # oblique line, ||f||=5 and diag(R)=(9/25,16/25), giving radii(3,4)/7.
    projector = ((F(9, 25), F(12, 25)), (F(12, 25), F(16, 25)))
    polynomial = ((_iv(0), _iv(0)), (_iv(0), _iv(0)))
    result = _interval_response(projector, polynomial, F(2), F(1, 7), (_iv(3), _iv(4)), _arithmetic())
    for bound, radius in zip(result, (F(3, 7), F(4, 7)), strict=True):
        _contains(bound, radius)
        _contains(bound, -radius)
        assert bound.hi < float(radius)+1e-15


@pytest.mark.parametrize("strict,sticky", [(False, False), (True, False), (False, True), (True, True)])
def test_helpers_need_no_backend_solve_seal_or_provenance_changes(strict, sticky, monkeypatch):
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)

    def forbidden(*args, **kwargs):
        pytest.fail("algebraic helper called a backend, solve, or certificate operation")

    for name in ("exp_iv", "ln_iv", "require_rigorous_backend", "set_strict_backend"):
        monkeypatch.setattr(transcend, name, forbidden)
    linear = import_module("omnibias.core._rational_linear")
    monkeypatch.setattr(linear, "solve_affine", forbidden)
    monkeypatch.setattr(linear, "factor_positive_definite", forbidden)
    certificate = import_module("omnibias.core.proof.certificate")
    monkeypatch.setattr(certificate, "make_certificate", forbidden)
    monkeypatch.setattr(certificate, "verify_certificate_digest", forbidden)
    probabilities = (_iv(F(1, 2)),)*2
    centers = ((F(0),), (F(1),))
    forcing = _bias_covariance(probabilities, centers, (F(0), F(1)), _arithmetic())
    assert _projected_mean(probabilities, centers, _arithmetic()) == (_iv(F(1, 2)),)
    result = _interval_response(((F(1),),), ((_iv(1),),), F(1), F(1, 2), forcing, _arithmetic())
    _contains(result[0], F(1, 4))
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used() is sticky


@pytest.mark.parametrize("operation", ["mean", "covariance", "response"])
def test_shared_work_meter_is_cumulative_and_does_not_reset_storage(operation, monkeypatch):
    a = _arithmetic()

    def forbidden(*args, **kwargs):
        pytest.fail("internal helper performed a separate storage reservation")

    monkeypatch.setattr(a, "_entries", forbidden)
    previous = 0
    for _ in range(3):
        if operation == "mean":
            _projected_mean((_iv(F(1, 2)),)*2, ((F(0),), (F(1),)), a)
        elif operation == "covariance":
            _bias_covariance((_iv(F(1, 2)),)*2, ((F(0),), (F(1),)), (F(0), F(1)), a)
        else:
            _interval_response(((F(1),),), ((_iv(1),),), F(1), F(1, 2), (_iv(1),), a)
        assert a.used > previous
        previous = a.used


@pytest.mark.parametrize("operation", ["mean", "covariance", "response"])
def test_low_work_and_coefficient_budgets_refuse(operation):
    for limits in ({"max_elimination_work": 1}, {"max_coefficient_bits": 8}):
        a = _arithmetic(**limits)
        with pytest.raises(ValueError):
            if operation == "mean":
                _projected_mean((_iv(F(1, 3)), _iv(F(2, 3))), ((F(0),), (F(1),)), a)
            elif operation == "covariance":
                _bias_covariance((_iv(F(1, 3)), _iv(F(2, 3))), ((F(0),), (F(1),)), (F(0), F(1)), a)
            else:
                _interval_response(((F(1),),), ((_iv(1),),), F(1), F(1, 3), (_iv(F(1, 3)),), a)


@pytest.mark.parametrize("centers", [(), ((),), ((F(1),),), ((F(0),), (F(1), F(2)))])
def test_malformed_projected_center_shapes_refuse(centers):
    with pytest.raises(ValueError):
        _projected_mean((_iv(1),)*len(centers), centers, _arithmetic())


@pytest.mark.parametrize("beta", [(F(0),), (False, F(1)), (0.0, 1.0)])
def test_bias_direction_shape_and_exact_types_refuse(beta):
    with pytest.raises((TypeError, ValueError)):
        _bias_covariance((_iv(F(1, 2)),)*2, ((F(0),), (F(1),)), beta, _arithmetic())


def test_nonfinite_arithmetic_refuses_without_changing_provenance():
    a = _arithmetic()
    state = transcend.strict_backend(), transcend.libm_fallback_used()
    with pytest.raises(ArithmeticError):
        _projected_mean((_iv(F(1, 2)),)*2, ((F(0),), (F(10**400),)), a)
    assert state == (transcend.strict_backend(), transcend.libm_fallback_used())
