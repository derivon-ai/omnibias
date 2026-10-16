# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Analytic-disk, genuine inverse-tail and exact resource-contract checks."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core.gaussian_bias_jet import gaussian_bias_inverse_series
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_bias_analytic import (
    GaussianBiasSeriesCertificate,
    _certify_bias_series,
    _enclose,
    _tail_bound,
    certify_gaussian_bias_inverse_series,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap


def _series(order=7, *, gauge=0, shift=0):
    model = GaussianHeatMap(((shift-1,), (shift+1,)), (gauge+F(1, 4),)*2)
    targets = ((-1,),)+((0,),)*(order-1) if order else ()
    biases = ((gauge, gauge+2),)+((gauge, gauge),)*(order-1) if order else ()
    return gaussian_bias_inverse_series(model, (shift,), targets, biases)


def _certificate(order=7):
    result = certify_gaussian_bias_inverse_series(_series(order), parameter_radius=F(1, 16), spatial_radius=F(1, 2))
    assert result is not None
    return result


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def test_bound_fields_replay_the_pinned_rational_formulas():
    certificate = _certificate()
    assert certificate.parameter_radius == F(1, 16) and certificate.spatial_radius == F(1, 2)
    assert certificate.diameter_bound == 2
    assert certificate.target_growth_bound == F(1, 16)
    assert certificate.bias_growth_bound == F(1, 8)
    assert certificate.phase_half_width_bound == F(5, 16)
    assert certificate.complex_contraction_bound == F(128, 231)
    assert certificate.forcing_bound == F(487, 3696)
    assert certificate.available_displacement == F(103, 462)
    assert certificate.forcing_bound <= certificate.available_displacement
    assert certificate.tail_bound(0) == 0
    assert certificate.tail_bound(F(1, 32)) == F(1, 256)
    assert certificate.certificate["payload"]["bias_coefficient_ranges"] == [[2, 1]]+[[0, 1]]*6


def test_independent_real_inverse_roots_and_complex_tail_are_enclosed():
    mp = pytest.importorskip("mpmath")
    certificates = [_certificate(order) for order in (3, 7, 11)]
    with mp.workdps(90):
        for sign in (-1, 1):
            parameter = sign*F(1, 64)
            t = _mp(mp, parameter)
            root = mp.findroot(lambda x, t=t: x+t-mp.tanh(x/2+t), (0, -t**3))
            previous = None
            for certificate in certificates:
                enclosure = certificate.enclose(parameter)[0]
                assert mp.mpf(enclosure.lo) <= root <= mp.mpf(enclosure.hi)
                error = abs(root-_mp(mp, certificate.series.evaluate(parameter)[0]))
                tail = certificate.tail_bound(abs(parameter))
                assert error <= _mp(mp, tail)
                if previous is not None:
                    assert tail == previous/256
                previous = tail
        certificate = certificates[1]
        t = _mp(mp, certificate.parameter_radius)*(1+mp.j)/8
        root = mp.findroot(lambda x: x+t-mp.tanh(x/2+t), (0, -t**3))
        polynomial = sum(_mp(mp, row[0])*t**n for n, row in enumerate(certificate.series.coefficients))
        assert abs(t) < _mp(mp, certificate.parameter_radius/4)
        assert abs(root-polynomial) <= _mp(mp, certificate.tail_bound(certificate.parameter_radius/4))
        assert abs(root) < _mp(mp, certificate.spatial_radius)


def test_coupled_original_gaussian_bias_and_target_paths_have_valid_enclosures():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (1, 0), (0, 1)), (0, F(1, 4), F(1, 4)))
    targets = ((F(1, 2), F(-1, 3)), (F(1, 4), F(1, 5)))+((0, 0),)*4
    biases = ((0, 1, -1), (0, F(1, 3), F(1, 2)))+((0, 0, 0),)*4
    series = gaussian_bias_inverse_series(model, (0, 0), targets, biases)
    certificate = certify_gaussian_bias_inverse_series(series, parameter_radius=F(1, 16), spatial_radius=F(1, 2))
    assert certificate is not None and certificate.diameter_bound**2 >= 2
    with mp.workdps(90):
        for sign in (-1, 1):
            parameter = sign*F(1, 64)
            t = _mp(mp, parameter)
            target = [sum(_mp(mp, row[i])*t**n for n, row in enumerate(series.target_coefficients)) for i in range(2)]
            bias = [sum(_mp(mp, row[j])*t**n for n, row in enumerate(series.bias_coefficients)) for j in range(3)]

            def residual(x, y, target=target, bias=bias):
                weights = [mp.exp(bias[j]-((x-center[0])**2+(y-center[1])**2)/4) for j, center in enumerate(model.centers)]
                total = sum(weights)
                return x-weights[1]/total-target[0], y-weights[2]/total-target[1]

            root = mp.findroot(residual, (t, -t))
            box = certificate.enclose(parameter)
            for interval, value in zip(box, root, strict=True):
                assert mp.mpf(interval.lo) <= value <= mp.mpf(interval.hi)


def test_bias_ranges_are_summed_coefficientwise_without_complex_cancellation():
    model = GaussianHeatMap(((-1,), (1,)), (F(1, 4),)*2)
    series = gaussian_bias_inverse_series(model, (0,), ((0,), (0,)), ((0, 2), (0, -2)))
    result = certify_gaussian_bias_inverse_series(series, parameter_radius=F(1, 16), spatial_radius=F(1, 2))
    assert result is not None
    assert result.bias_growth_bound == F(17, 128)
    assert result.target_growth_bound == 0
    assert result.certificate["payload"]["bias_coefficient_ranges"] == [[2, 1], [2, 1]]


def test_exact_common_bias_and_spatial_gauges_preserve_all_analytic_bounds():
    plain, shifted = _certificate(), certify_gaussian_bias_inverse_series(
        _series(gauge=2**3000, shift=2**2500), parameter_radius=F(1, 16), spatial_radius=F(1, 2))
    assert shifted is not None
    for field in ("diameter_bound", "target_growth_bound", "bias_growth_bound", "phase_half_width_bound",
                  "complex_contraction_bound", "forcing_bound", "available_displacement"):
        assert getattr(plain, field) == getattr(shifted, field)
    assert plain.tail_bound(F(1, 32)) == shifted.tail_bound(F(1, 32))


def test_inclusive_self_map_boundary_and_no_entire_shortcut():
    model = GaussianHeatMap(((3,),), (0,), time=2)
    series = gaussian_bias_inverse_series(model, (5,), ((1,),), ((7,),))
    result = certify_gaussian_bias_inverse_series(series, parameter_radius=F(1, 4), spatial_radius=F(1, 2))
    assert result is not None
    assert result.forcing_bound == result.available_displacement == F(1, 2)
    assert result.diameter_bound == result.phase_half_width_bound == result.complex_contraction_bound == 0
    assert result.parameter_radius == F(1, 4) and result.spatial_radius == F(1, 2)
    assert result.tail_bound(F(1, 8)) == F(1, 4) > 0
    assert result.enclose(F(1, 8))[0].contains(F(21, 4))
    assert certify_gaussian_bias_inverse_series(series, parameter_radius=F(1, 4), spatial_radius=F(1, 3)) is None


def test_zero_diameter_still_checks_partition_phase_under_bias_changes():
    model = GaussianHeatMap(((0,), (0,)), (0, 0))
    series = gaussian_bias_inverse_series(model, (0,), ((0,),), ((0, 10),))
    result = certify_gaussian_bias_inverse_series(series, parameter_radius=F(1, 16), spatial_radius=F(1, 10))
    assert result is not None and result.diameter_bound == 0
    assert result.bias_growth_bound == F(5, 8) and result.phase_half_width_bound == F(5, 16)
    assert result.forcing_bound == 0 and result.complex_contraction_bound == 0
    assert certify_gaussian_bias_inverse_series(series, parameter_radius=F(1, 5), spatial_radius=1) is None  # eta²=1
    assert certify_gaussian_bias_inverse_series(series, parameter_radius=F(1, 4), spatial_radius=1) is None


def test_constant_target_and_order_zero_still_have_finite_domains():
    constant = _series(order=0)
    result = certify_gaussian_bias_inverse_series(constant, parameter_radius=3, spatial_radius=1)
    assert result is not None and result.parameter_radius == 3 and result.spatial_radius == 1
    assert result.target_growth_bound == result.bias_growth_bound == result.forcing_bound == 0
    assert result.tail_bound(1) == F(1, 2)
    # A constant path does not bypass the sufficient spatial contraction test.
    assert certify_gaussian_bias_inverse_series(constant, parameter_radius=3, spatial_radius=2) is None
    assert certify_gaussian_bias_inverse_series(_series(), parameter_radius=F(1, 8), spatial_radius=F(1, 2)) is None


def test_exact_outward_conversion_and_unrepresentable_endpoint_refusal():
    model = GaussianHeatMap(((F(1, 3),),), (0,), time=F(2, 3))
    series = gaussian_bias_inverse_series(model, (F(1, 3),), (), ())
    result = certify_gaussian_bias_inverse_series(series, parameter_radius=1, spatial_radius=1)
    assert result is not None
    interval = result.enclose(0)[0]
    assert F(interval.lo) <= F(1, 3) <= F(interval.hi) and interval.lo < interval.hi
    huge = gaussian_bias_inverse_series(GaussianHeatMap(((0,),), (0,)), (2**2000,), (), ())
    huge_certificate = certify_gaussian_bias_inverse_series(huge, parameter_radius=1, spatial_radius=1)
    assert huge_certificate is not None
    with pytest.raises(ArithmeticError, match="binary64"):
        huge_certificate.enclose(0)


def test_factory_only_record_complete_binding_and_digest_checked_methods():
    result = _certificate()
    with pytest.raises(TypeError):
        GaussianBiasSeriesCertificate()
    with pytest.raises(FrozenInstanceError):
        result.parameter_radius = F(1)
    assert verify_certificate_digest(result.certificate) and schema_errors_v1(result.certificate) == []
    assert certificate_is_unconditional(result.certificate)
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    payload = result.certificate["payload"]
    assert payload["bias_coefficients"] == [[list((value.numerator, value.denominator)) for value in row]
                                            for row in result.series.bias_coefficients]
    assert len(payload["posterior_coefficients"]) == result.series.order+1
    intact = deepcopy(result.certificate)
    result.certificate["payload"]["coefficients"][3] = [[0, 1]]
    assert not verify_certificate_digest(result.certificate)
    with pytest.raises(ValueError, match="damaged"):
        result.tail_bound(F(1, 64))
    with pytest.raises(ValueError, match="damaged"):
        result.enclose(F(1, 64))
    assert verify_certificate_digest(intact)


@pytest.mark.parametrize("name", ["parameter_radius", "spatial_radius"])
@pytest.mark.parametrize("value,error", [(None, TypeError), (True, TypeError), (0.5, TypeError), (complex(1), TypeError), (0, ValueError), (-1, ValueError)])
def test_radius_validation_precedes_preflight(name, value, error, monkeypatch):
    series = _series()
    kwargs = {"parameter_radius": F(1, 16), "spatial_radius": F(1, 2), name: value}
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_: pytest.fail("invalid radius reached preflight"))
    with pytest.raises(error):
        certify_gaussian_bias_inverse_series(series, **kwargs)


def test_invalid_series_raises():
    with pytest.raises(TypeError):
        certify_gaussian_bias_inverse_series(None, parameter_radius=1, spatial_radius=1)


@pytest.mark.parametrize("method,value,error", [("tail_bound", -1, ValueError), ("tail_bound", F(1, 16), ValueError),
    ("tail_bound", 1, ValueError), ("tail_bound", True, TypeError), ("tail_bound", 0.0, TypeError),
    ("enclose", F(1, 16), ValueError), ("enclose", F(-1, 16), ValueError), ("enclose", 1, ValueError),
    ("enclose", False, TypeError), ("enclose", 0.0, TypeError), ("enclose", complex(0), TypeError)])
def test_evaluation_domain_validation_precedes_preflight(method, value, error, monkeypatch):
    certificate = _certificate()
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_: pytest.fail("invalid evaluation reached preflight"))
    with pytest.raises(error):
        getattr(certificate, method)(value)


@pytest.mark.parametrize("sticky", [False, True])
def test_exact_proof_needs_no_exponentials_or_factorization_and_preserves_state(sticky, monkeypatch):
    series = _series()

    def fail(*args, **kwargs):
        pytest.fail("analytic rational certificate attempted a transcendental or solve")

    for name, functions in (("omnibias.core._rational_linear", ("factor_positive_definite", "solve_affine")),
                            ("omnibias.core.verified.inverse_gaussian_heat", ("inverse_gaussian_heat", "certify_gaussian_inverse"))):
        module = import_module(name)
        for function in functions:
            monkeypatch.setattr(module, function, fail)
    monkeypatch.setattr(transcend, "exp_iv", fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    certificate = certify_gaussian_bias_inverse_series(series, parameter_radius=F(1, 16), spatial_radius=F(1, 2))
    assert certificate is not None and certificate.certificate["meta"]["transcend_backend"] == "not_used"
    certificate.enclose(F(1, 64))
    assert transcend.libm_fallback_used() is sticky and not transcend.strict_backend()


def test_strict_sealing_policy_is_not_bypassed(monkeypatch):
    series = _series()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_bias_inverse_series(series, parameter_radius=F(1, 16), spatial_radius=F(1, 2))
    assert transcend.strict_backend()


@pytest.mark.parametrize("limits", [RidgeBudget(max_matrix_entries=300), RidgeBudget(max_elimination_work=800)])
def test_analytic_aggregate_preflight_can_refuse_after_exact_series_construction(limits, monkeypatch):
    series = gaussian_bias_inverse_series(GaussianHeatMap(((0,),), (0,)), (0,), (), (), budget=limits)
    module = import_module("omnibias.core.verified.gaussian_bias_analytic")
    monkeypatch.setattr(module, "_sqrt_upper", lambda *_: pytest.fail("preflight missed analytic arithmetic"))
    assert certify_gaussian_bias_inverse_series(series, parameter_radius=1, spatial_radius=1) is None


def test_input_bit_growth_and_unrepresentable_diameter_return_inconclusive():
    assert certify_gaussian_bias_inverse_series(_series(), parameter_radius=2**5000, spatial_radius=1) is None
    scale = 2**1500
    model = GaussianHeatMap(((0,), (scale,)), (0, F(1, 4)), viscosity=scale, time=scale)
    series = gaussian_bias_inverse_series(model, (0,), (), ())
    assert certify_gaussian_bias_inverse_series(series, parameter_radius=1, spatial_radius=1) is None


def test_private_build_and_evaluation_share_precharged_resource_meters():
    series = _series()
    arithmetic = _Arithmetic(series.budget)
    arithmetic._work(series.budget.max_elimination_work-1)
    assert _certify_bias_series(series, F(1, 16), F(1, 2), arithmetic) is None
    assert arithmetic.used > series.budget.max_elimination_work
    certificate = _certificate()
    arithmetic = _Arithmetic(series.budget)
    arithmetic._work(100)
    tail = _tail_bound(certificate, F(1, 64), arithmetic)
    after_tail = arithmetic.used
    box = _enclose(certificate, F(1, 64), arithmetic)
    assert tail == certificate.tail_bound(F(1, 64)) and box == certificate.enclose(F(1, 64))
    assert 100 < after_tail < arithmetic.used
    arithmetic._work(series.budget.max_elimination_work-arithmetic.used)
    with pytest.raises(ValueError, match="work"):
        _enclose(certificate, F(1, 64), arithmetic)


def test_public_methods_preflight_before_digest_processing(monkeypatch):
    certificate = _certificate()
    module = import_module("omnibias.core.verified.gaussian_bias_analytic")

    def exhausted(*args):
        raise ValueError("test evaluation budget exhausted")

    monkeypatch.setattr(RidgeBudget, "_entries", exhausted)
    monkeypatch.setattr(module, "_intact", lambda *_: pytest.fail("preflight missed certificate integrity processing"))
    for method in (certificate.tail_bound, certificate.enclose):
        with pytest.raises(ValueError, match="budget"):
            method(F(1, 64))
