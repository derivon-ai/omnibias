# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Contraction inverse bounds, exact norm rounding, and refusal semantics."""

import math
from copy import deepcopy
from fractions import Fraction

import pytest
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.inverse_gaussian_heat import (
    _sqrt_upper,
    certify_gaussian_inverse,
    inverse_gaussian_heat,
)


def _binary(time=1):
    return GaussianHeatMap(((-1,), (1,)), (0, 0), time=time)


@pytest.mark.parametrize("value", [Fraction(0), Fraction(1), Fraction(2), Fraction(1, 3),
                                  Fraction(2)**-2200, Fraction(2)**-2148,
                                  Fraction(2)**-2000, Fraction(2)**2000,
                                  Fraction.from_float(1.7976931348623157e308)**2])
def test_exact_square_root_upper_bound_extreme_scales(value):
    bound = _sqrt_upper(value)
    assert math.isfinite(bound) and Fraction(bound)**2 >= value
    if bound > 0:
        # The exact integer significand rounds upward by at most two float steps.
        previous = math.nextafter(math.nextafter(bound, -math.inf), -math.inf)
        assert previous <= 0 or Fraction(previous)**2 < value


def test_square_root_refuses_negative_and_nonfinite_output():
    with pytest.raises(ValueError):
        _sqrt_upper(Fraction(-1))
    with pytest.raises(ArithmeticError):
        _sqrt_upper(Fraction(2)**3000)


def test_square_root_scaling_obeys_exact_intermediate_budget():
    with pytest.raises(ValueError, match="budget"):
        _sqrt_upper(Fraction(1), RidgeBudget(max_coefficient_bits=64))
    bound = _sqrt_upper(Fraction(2), RidgeBudget(max_coefficient_bits=128))
    assert Fraction(bound)**2 >= 2


def test_any_point_certifies_a_coarse_global_inverse():
    pytest.importorskip("mpmath")
    certificate = certify_gaussian_inverse(_binary(), (0,), (4.0,))
    assert certificate is not None
    assert certificate.enclosure[0].contains(0.0)
    assert certificate.enclosure[0].lo >= -1 and certificate.enclosure[0].hi <= 1
    assert certificate.contraction_bound == Fraction(1, 2)
    assert certificate.scaled_error_bound >= 4
    assert certificate_is_unconditional(certificate.certificate)
    assert schema_errors_v1(certificate.certificate) == []
    assert "theorem_prover_verified" not in certificate.certificate["honesty"]
    modified = deepcopy(certificate.certificate)
    modified["payload"]["time"] = [99, 1]
    assert not verify_certificate_digest(modified)


def test_iteration_encloses_independent_root_and_meets_tolerance():
    mp = pytest.importorskip("mpmath")
    model = _binary(time=2)
    result = inverse_gaussian_heat(model, (Fraction(1, 3),), absolute_tolerance=1e-11)
    assert result.status == "certified" and result.tolerance_met
    assert result.certificate is not None and 0 < result.iterations < 128
    with mp.workdps(80):
        root = mp.findroot(lambda x: (x-mp.tanh(x/4))/2-mp.mpf(1)/3, 1)
        bound = result.certificate.enclosure[0]
        assert mp.mpf(bound.lo) <= root <= mp.mpf(bound.hi)
        error = abs(root-2*mp.mpf(result.scaled_point[0]))
        assert error <= result.certificate.error_bound <= 1e-11


def test_coarse_certification_is_distinguished_from_requested_accuracy():
    pytest.importorskip("mpmath")
    result = inverse_gaussian_heat(_binary(), (1,), max_iterations=0)
    assert result.status == "certified" and not result.tolerance_met
    assert result.certificate is not None and result.iterations == 0


@pytest.mark.parametrize("time", [Fraction(1, 2), Fraction(1, 4)])
def test_outside_sufficient_regime_is_not_infeasibility(time, monkeypatch):
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    model = _binary(time=time)
    result = inverse_gaussian_heat(model, (0,))
    assert result.status == "outside_contraction_regime"
    assert result.certificate is None and not result.tolerance_met
    assert certify_gaussian_inverse(model, (0,), (0.0,)) is None


def test_rank_deficient_single_center_has_analytic_inverse():
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(((2, -3),), (10**100,), time=Fraction(3, 2))
    result = inverse_gaussian_heat(model, (Fraction(1, 3), 2))
    assert result.status == "certified" and result.tolerance_met
    assert result.certificate is not None
    assert result.certificate.contraction_bound == 0
    assert result.certificate.enclosure[0].contains(2.5)
    assert result.certificate.enclosure[1].contains(0.0)


@pytest.mark.parametrize("sticky", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_certificate_preserves_transcendental_provenance(monkeypatch, sticky, strict):
    pytest.importorskip("mpmath")
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    certificate = certify_gaussian_inverse(_binary(), (0,), (0.0,))
    assert certificate is not None and certificate_is_unconditional(certificate.certificate)
    assert transcend.libm_fallback_used() is sticky
    assert transcend.strict_backend() is strict


def test_missing_rigorous_backend_raises(monkeypatch):
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        inverse_gaussian_heat(_binary(), (0,))


@pytest.mark.parametrize("target, point", [((0.0,), (0.0,)), ((True,), (0.0,)),
                                          ((0,), (True,)), ((0,), (Fraction(1, 3),))])
def test_ambiguous_input_types_raise(target, point):
    with pytest.raises(TypeError):
        certify_gaussian_inverse(_binary(), target, point)


@pytest.mark.parametrize("target, point", [((0, 1), (0.0,)), ((0,), ()),
                                          ((0,), (math.inf,)), ((0,), (math.nan,)),
                                          ((0,), (2**60+1,))])
def test_invalid_input_values_raise(target, point):
    with pytest.raises(ValueError):
        certify_gaussian_inverse(_binary(), target, point)


@pytest.mark.parametrize("options, error", [({"max_iterations": True}, TypeError),
                                            ({"max_iterations": -1}, ValueError),
                                            ({"absolute_tolerance": 0.0}, ValueError),
                                            ({"absolute_tolerance": math.inf}, ValueError)])
def test_invalid_iteration_controls(options, error):
    with pytest.raises(error):
        inverse_gaussian_heat(_binary(), (0,), **options)


def test_budget_is_cumulative_and_keeps_last_valid_certificate():
    pytest.importorskip("mpmath")
    # Enough to construct/evaluate once, but too little for a long solve.
    budget = RidgeBudget(max_elimination_work=12000)
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), time=Fraction(3, 5), budget=budget)
    result = inverse_gaussian_heat(model, (1,), max_iterations=10000, absolute_tolerance=1e-100)
    assert result.iterations < 10000
    assert result.status == "certified" and result.certificate is not None
    assert not result.tolerance_met


def test_overflow_is_inconclusive_not_a_false_enclosure():
    pytest.importorskip("mpmath")
    result = certify_gaussian_inverse(_binary(), (10**300,), (-1e308,))
    assert result is None
