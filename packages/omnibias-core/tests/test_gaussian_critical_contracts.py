# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Critical Gaussian model binding, exact cube roots and proof boundaries."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core._rational_linear import RationalLDLT
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_critical import (
    GaussianCriticalChart,
    _cube_root_interval,
    certify_gaussian_critical_chart,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap


def _model(*, budget=None, bias=0, shift=0):
    return GaussianHeatMap(((shift-1,), (shift+1,)), (0, bias), viscosity=F(1, 2), budget=budget)


def _chart(**kwargs):
    result = certify_gaussian_critical_chart(_model(**kwargs))
    assert result is not None
    return result


def test_exact_proof_constants_and_model_binding():
    chart = _chart(bias=F(1, 3))
    assert chart.model.contraction_bound == 1
    assert chart.midpoint == (0,) and chart.half_difference == (1,)
    assert chart.squared_half_distance == 1 and chart.orthogonal_projector == ((0,),)
    assert chart.bias_contrast == F(1, 3)
    assert chart.complex_contraction_bound == F(37321, 120704)
    assert chart.parameter_radius == F(1, 4) and chart.shape_radius == F(1, 8)
    payload = chart.certificate["payload"]
    terms = tuple(F(*pair) for pair in payload["contraction_terms"])
    assert sum(terms) == chart.complex_contraction_bound
    assert F(*payload["base_defect_bound"]) == F(1, 90)
    assert F(*payload["available_displacement"]) == (1-chart.complex_contraction_bound)*F(1, 8)
    assert chart.target(0) == (F(-1, 6),)


@pytest.mark.parametrize("model", [
    GaussianHeatMap(((0,),), (0,)),
    GaussianHeatMap(((0,), (0,)), (0, 1)),
    GaussianHeatMap(((-1,), (0,), (1,)), (0, 0, 0), viscosity=F(1, 2)),
    GaussianHeatMap(((-1,), (1,)), (0, 0)),
    GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=F(1, 4)),
])
def test_unsupported_families_refuse_without_replacing_the_supplied_model(model):
    assert certify_gaussian_critical_chart(model) is None


@pytest.mark.parametrize("order,error", [(-1, ValueError), (True, TypeError), (1.0, TypeError), (None, TypeError)])
def test_invalid_order_raises_before_resource_preflight(order, error, monkeypatch):
    model = _model()

    def fail(*args):
        pytest.fail("invalid order reached preflight")

    monkeypatch.setattr(_Arithmetic, "_entries", fail)
    with pytest.raises(error):
        certify_gaussian_critical_chart(model, order=order)


def test_bad_model_and_direct_constructor_raise():
    with pytest.raises(TypeError):
        certify_gaussian_critical_chart(None)
    with pytest.raises(TypeError):
        GaussianCriticalChart()


@pytest.mark.parametrize("method", ["target", "enclose"])
@pytest.mark.parametrize("parameter,error", [(True, TypeError), (0.0, TypeError), (None, TypeError),
                                              (F(1, 4), ValueError), (F(-1, 4), ValueError)])
def test_parameter_queries_validate_the_strict_analytic_domain(method, parameter, error):
    chart = _chart()
    with pytest.raises(error):
        getattr(chart, method)(parameter)


@pytest.mark.parametrize("method", ["target", "enclose"])
@pytest.mark.parametrize("vector,error", [((), ValueError), ((0, 0), ValueError), ((0.0,), TypeError),
                                           ((False,), TypeError), ((1,), ValueError)])
def test_transverse_target_is_exact_and_orthogonal(method, vector, error):
    chart = _chart()
    with pytest.raises(error):
        getattr(chart, method)(F(1, 16), vector)


@pytest.mark.parametrize("value", [F(1, 1001), F(-1, 1001), F(1, 2**600), F(-1, 2**600), F(1, 512), F(-1, 512)])
@pytest.mark.parametrize("precision", [1, 8, 80])
def test_exact_cube_root_brackets_and_relative_width_at_extreme_scales(value, precision):
    lo, hi = _cube_root_interval(value, precision, _Arithmetic(RidgeBudget()))
    assert lo**3 <= value <= hi**3
    assert (hi-lo)**3 <= abs(value)*F(8, 2**(3*precision))
    if abs(value) == F(1, 512):
        assert lo == hi == (F(1, 8) if value > 0 else F(-1, 8))


def test_zero_target_needs_no_cube_root_precision_allocation():
    chart = _chart(budget=RidgeBudget(max_coefficient_bits=64))
    assert chart.enclose_target((0,))[0].lo <= 0 <= chart.enclose_target((0,))[0].hi
    assert chart.enclose_target((0,), precision=10**100)[0].contains(0)
    with pytest.raises(ValueError):
        chart.enclose_target((F(1, 3000),))


def test_low_precision_boundary_bracket_refuses_and_more_precision_recovers():
    chart = _chart()
    t = F(255, 1024)
    target = (t**3/3,)
    with pytest.raises(ValueError, match="bracket reaches"):
        chart.enclose_target(target, precision=1)
    result = chart.enclose_target(target, precision=16)
    assert result[0].lo < result[0].hi


@pytest.mark.parametrize("target,error", [((), ValueError), ((0, 0), ValueError), ((False,), TypeError),
                                          ((0.0,), TypeError), ((F(1, 192),), ValueError), ((F(-1, 192),), ValueError)])
def test_direct_target_query_validates_exact_metadata_and_domain(target, error):
    chart = _chart()
    with pytest.raises(error):
        chart.enclose_target(target)


@pytest.mark.parametrize("precision,error", [(True, TypeError), (80.0, TypeError), (None, TypeError), (0, ValueError), (-1, ValueError)])
def test_cube_root_precision_metadata_is_validated(precision, error):
    with pytest.raises(error):
        _chart().enclose_target((0,), precision=precision)


def test_common_absolute_coordinates_and_bias_gauges_leave_critical_geometry_exact():
    model = GaussianHeatMap(((2**1800-1,), (2**1800+1,)), (2**2500, 2**2500+F(1, 3)), viscosity=F(1, 2))
    chart = certify_gaussian_critical_chart(model)
    assert chart is not None and chart.half_difference == (1,) and chart.midpoint == (2**1800,)
    assert chart.bias_contrast == F(1, 3)
    assert chart.target(0) == (F(-1, 6),)
    with pytest.raises((ArithmeticError, ValueError, OverflowError)):
        chart.enclose(0)


def test_certificate_schema_snapshot_and_query_integrity():
    chart = _chart()
    assert verify_certificate_digest(chart.certificate)
    assert schema_errors_v1(chart.certificate) == [] and certificate_is_unconditional(chart.certificate)
    assert "theorem_prover_verified" not in chart.certificate["honesty"]
    with pytest.raises(FrozenInstanceError):
        chart.parameter_radius = F(1)
    saved = deepcopy(chart.certificate)
    chart.certificate["payload"]["parameter_radius"] = [1, 1]
    assert not verify_certificate_digest(chart.certificate) and verify_certificate_digest(saved)
    for method, argument in ((chart.enclose, F(1, 16)), (chart.target, F(1, 16)),
                             (chart.tail_bound, F(1, 16)), (chart.enclose_target, (0,))):
        with pytest.raises(ValueError, match="damaged"):
            method(argument)


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_factory_preflight_precedes_geometry_allocation(guard, monkeypatch):
    model = _model()

    def exhausted(*args):
        raise ValueError("resource exhausted")

    def fail(*args):
        pytest.fail("failed preflight reached exact geometry")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(_Arithmetic, "add", fail)
    assert certify_gaussian_critical_chart(model) is None


def test_series_geometry_and_proof_share_the_same_cumulative_meter(monkeypatch):
    model = _model()
    module = import_module("omnibias.core.verified.gaussian_critical")
    original = module._build_critical_series
    observed, child_counts = [], []

    def count(self, work):
        observed.append(work)
        assert work <= self.max_elimination_work

    def child(*args):
        child_counts.append(args[-1].used)
        result = original(*args)
        child_counts.append(args[-1].used)
        return result

    monkeypatch.setattr(RidgeBudget, "_work", count)
    monkeypatch.setattr(module, "_build_critical_series", child)
    result = certify_gaussian_critical_chart(model)
    assert result is not None
    assert observed[0] < child_counts[0] < child_counts[1] < observed[-1]
    assert all(a < b for a, b in zip(observed, observed[1:], strict=False))
    observed.clear()
    result.enclose_target((F(1, 3003),))
    assert all(a < b for a, b in zip(observed, observed[1:], strict=False))


def test_resource_and_rational_growth_failures_are_explicit():
    model = _model(budget=RidgeBudget(max_order=8))
    assert certify_gaussian_critical_chart(model, order=9) is None
    chart = _chart()
    with pytest.raises(ValueError):
        chart.enclose_target((F(1, 2**5000),))
    with pytest.raises(ValueError):
        chart.enclose_target((F(1, 3000),), precision=10000)
    with pytest.raises(ValueError):
        chart.tail_bound(F(1, 2**5000))


def test_factory_and_queries_need_no_numerical_solve_or_transcendental_backend(monkeypatch):
    model = _model()

    def fail(*args, **kwargs):
        pytest.fail("critical chart attempted a factorization, solve or exponential")

    monkeypatch.setattr(RationalLDLT, "solve", fail)
    monkeypatch.setattr(import_module("omnibias.core._rational_linear"), "factor_positive_definite", fail)
    monkeypatch.setattr(transcend, "exp_iv", fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    chart = certify_gaussian_critical_chart(model)
    assert chart is not None and chart.certificate["meta"]["transcend_backend"] == "not_used"
    chart.enclose(F(1, 16))
    chart.enclose_target((F(1, 3003),))
    assert transcend.libm_fallback_used() and not transcend.strict_backend()


@pytest.mark.parametrize("sticky", [False, True])
def test_strict_certificate_policy_preserves_sticky_history(sticky, monkeypatch):
    model = _model()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    if sticky:
        with pytest.raises(RuntimeError, match="libm_fallback"):
            certify_gaussian_critical_chart(model)
    else:
        assert certify_gaussian_critical_chart(model) is not None
    assert transcend.strict_backend() and transcend.libm_fallback_used() is sticky


def test_strict_certificate_policy_requires_optional_backend(monkeypatch):
    model = _model()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_critical_chart(model)
