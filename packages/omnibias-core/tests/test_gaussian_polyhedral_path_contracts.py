# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact branch signs retain structural zeros, binding, and budget contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_polyhedral_equilibrium import (
    certify_gaussian_polyhedral_equilibrium,
)
from omnibias.core.verified.gaussian_polyhedral_path import (
    GaussianPolyhedralPath,
    compile_gaussian_polyhedral_path,
)


def _equilibrium(*, selected=(), shift=0, budget=None, inequalities=((1,),), upper=None):
    model = GaussianHeatMap(((shift-1,), (shift+1,)), (F(1, 4),)*2, budget=budget)
    base = restrict_gaussian_heat(model, (), ())
    assert base is not None
    face = compile_gaussian_polyhedral_face(base, inequalities, (shift,) if upper is None else upper, selected)
    assert face is not None
    eq = certify_gaussian_polyhedral_equilibrium(face, (0,), (shift,))
    assert eq is not None
    return eq


@pytest.mark.parametrize("selected", [(), (0,)])
def test_common_bias_polynomial_is_an_exact_constant_path_identity(selected):
    eq = _equilibrium(selected=selected)
    gauge = 2**2500
    result = compile_gaussian_polyhedral_path(eq, ((0,), (0,), (0,)), ((gauge, gauge), (3, 3), (-2, -2)))
    assert result is not None and result.status == "valid"
    assert result.coefficients == ((0,),)*4
    assert result.posterior_coefficients == ((F(1, 2),)*2, (0, 0), (0, 0), (0, 0))
    assert result.eventual_active_indices == (0,)
    assert result.sign_tests[0].verdict == "zero"
    assert result.sign_tests[0].identity is not None


def test_zero_prefix_without_identity_remains_inconclusive():
    eq = _equilibrium()
    result = compile_gaussian_polyhedral_path(eq, ((-1,), (0,)), ((0, 2), (0, 0)))
    assert result is not None and result.coefficients == ((0,),)*3
    assert result.status == "inconclusive" and result.eventual_active_indices is None
    sign = result.sign_tests[0]
    assert sign.verdict == "inconclusive" and sign.identity is None and sign.leading_order is None
    assert result.certificate["payload"]["radius"] is None


def test_constant_normal_moment_certifies_zero_multiplier_while_tangent_moves():
    model = GaussianHeatMap(((-1, 0), (1, 0)), (F(1, 4),)*2)
    base = restrict_gaussian_heat(model, (), ())
    assert base is not None
    face = compile_gaussian_polyhedral_face(base, ((0, 1),), (0,), (0,))
    assert face is not None
    eq = certify_gaussian_polyhedral_equilibrium(face, (0, 0), (0, 0))
    assert eq is not None
    result = compile_gaussian_polyhedral_path(eq, ((1, 0), (0, 0), (0, 0)), ((0, 0),)*3)
    assert result is not None and result.status == "valid"
    assert result.coefficients[1] == (2, 0) and result.coefficients[3] == (F(-2, 3), 0)
    assert result.inequality_multiplier_coefficients == ((0,),)*4
    assert result.sign_tests[0].identity == "normal center contrasts and target increments vanish"


def test_structurally_tight_omitted_rows_are_in_eventual_active_set():
    eq = _equilibrium(selected=(0,), inequalities=((1,), (2,), (0,)), upper=(0, 0, 0))
    result = compile_gaussian_polyhedral_path(eq, ((1,),), ((0, 0),))
    assert result is not None and result.status == "valid"
    assert result.eventual_active_indices == (0, 1, 2)
    assert tuple(sign.verdict for sign in result.sign_tests) == ("positive", "zero", "zero")
    assert result.inequality_multiplier_coefficients[1] == (1, 0, 0)


def test_tiny_positive_leading_coefficient_is_not_rounded_to_zero():
    model = GaussianHeatMap(((0,),), (0,))
    base = restrict_gaussian_heat(model, (), ())
    assert base is not None
    face = compile_gaussian_polyhedral_face(base, ((1,),), (0,))
    assert face is not None
    eq = certify_gaussian_polyhedral_equilibrium(face, (0,), (0,))
    assert eq is not None
    epsilon = F(1, 2**2000)
    result = compile_gaussian_polyhedral_path(eq, ((-epsilon,),), ((0,),))
    assert result is not None and result.status == "valid"
    assert result.sign_tests[0].leading_coefficient == epsilon
    assert result.sign_tests[0].leading_order == 1 and result.eventual_active_indices == ()


def test_ambient_nonconvexity_does_not_replace_base_tangent_bound():
    model = GaussianHeatMap(((-1, 0), (1, 4)), (F(1, 4), F(17, 4)))
    assert model.contraction_bound > 1
    base = restrict_gaussian_heat(model, ((0, 1),), (0,))
    assert base is not None and base.model.contraction_bound == F(1, 2)
    face = compile_gaussian_polyhedral_face(base, (), ())
    assert face is not None
    eq = certify_gaussian_polyhedral_equilibrium(face, (0, 0), (0, 0))
    assert eq is not None
    result = compile_gaussian_polyhedral_path(eq, ((1, 2),), ((0, 0),))
    assert result is not None and result.status == "valid"
    assert result.coefficients == ((0, 0), (2, 0))
    assert result.equality_multiplier_coefficients == ((2,), (4,))
    assert result.posterior_coefficients == ((F(1, 2),)*2, (F(-1, 2), F(1, 2)))


def test_huge_spatial_and_bias_gauges_preserve_path_jets():
    target = ((-1,), (0,), (0,))
    original = compile_gaussian_polyhedral_path(_equilibrium(), target, ((0, 2), (0, 0), (0, 0)))
    shift, gauge = 2**1800, 2**2500
    translated = compile_gaussian_polyhedral_path(_equilibrium(shift=shift), target,
                                                 ((gauge, gauge+2), (gauge, gauge), (-gauge, -gauge)))
    assert original is not None and translated is not None
    assert translated.coefficients[0] == (shift,)
    assert translated.coefficients[1:] == original.coefficients[1:]
    for field in ("posterior_coefficients", "slack_coefficients", "inequality_multiplier_coefficients", "status", "sign_tests"):
        assert getattr(original, field) == getattr(translated, field)


def test_empty_order_describes_constant_parameter_polynomials():
    result = compile_gaussian_polyhedral_path(_equilibrium(), (), ())
    assert result is not None and result.order == 0 and result.status == "valid"
    assert result.eventual_active_indices == (0,)


@pytest.mark.parametrize("dependency", ["equilibrium", "face", "base", "geometry", "restriction"])
def test_tampered_dependencies_refuse_before_exact_data(dependency, monkeypatch):
    eq = _equilibrium(selected=(0,))
    record = {"equilibrium": eq, "face": eq.face, "base": eq.face.base_restriction,
              "geometry": eq.face.geometry, "restriction": eq.face.geometry.restriction}[dependency]
    record.certificate["payload"]["damaged"] = object()

    def fail(*args):
        pytest.fail("damaged evidence reached exact arithmetic")

    monkeypatch.setattr(_Arithmetic, "exact", fail)
    assert compile_gaussian_polyhedral_path(eq, ((1,),), ((0, 0),)) is None


@pytest.mark.parametrize("which,bad,error", [
    (0, None, TypeError), (1, None, TypeError), (2, None, TypeError),
    (1, "0", TypeError), (2, "0", TypeError), (1, (), ValueError), (2, (), ValueError),
    (1, ((1, 2),), ValueError), (2, ((0,),), ValueError),
    (1, ((1.0,),), TypeError), (2, ((False, 0),), TypeError),
    (1, (None,), TypeError), (2, ("00",), TypeError),
])
def test_invalid_input_raises(which, bad, error):
    args = [_equilibrium(), ((1,),), ((0, 0),)]
    args[which] = bad
    with pytest.raises(error):
        compile_gaussian_polyhedral_path(*args)


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_resource_preflight_precedes_exact_allocation(guard, monkeypatch):
    eq = _equilibrium()

    def exhausted(*args):
        raise ValueError("resource exhausted")

    def fail(*args):
        pytest.fail("failed preflight reached exact data")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(_Arithmetic, "exact", fail)
    assert compile_gaussian_polyhedral_path(eq, ((1,),), ((0, 0),)) is None


def test_series_and_certificate_use_one_cumulative_work_meter(monkeypatch):
    eq = _equilibrium()
    observed, builder_counts = [], []
    module = import_module("omnibias.core.verified.gaussian_polyhedral_path")
    original = module._build_equal_logits

    def count(self, amount):
        observed.append(amount)
        assert amount <= self.max_elimination_work

    def builder(*args):
        arithmetic = args[-1]
        builder_counts.append(arithmetic.used)
        result = original(*args)
        builder_counts.append(arithmetic.used)
        return result

    monkeypatch.setattr(RidgeBudget, "_work", count)
    monkeypatch.setattr(module, "_build_equal_logits", builder)
    result = compile_gaussian_polyhedral_path(eq, ((-1,), (0,), (0,)), ((0, 2), (0, 0), (0, 0)))
    assert result is not None
    assert observed[0] < builder_counts[0] < builder_counts[1] < observed[-1]
    assert all(left < right for left, right in zip(observed, observed[1:], strict=False))


def test_order_and_coefficient_growth_refuse():
    eq = _equilibrium(budget=RidgeBudget(max_order=2, max_coefficient_bits=64))
    assert compile_gaussian_polyhedral_path(eq, ((0,),)*3, ((0, 0),)*3) is None
    assert compile_gaussian_polyhedral_path(eq, ((2**100,),), ((0, 0),)) is None
    assert compile_gaussian_polyhedral_path(eq, ((0,),), ((0, 2**100),)) is None


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("sticky", [False, True])
def test_exact_path_preserves_backend_flags(strict, sticky, monkeypatch):
    eq = _equilibrium()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    if strict and sticky:
        with pytest.raises(RuntimeError, match="libm_fallback"):
            compile_gaussian_polyhedral_path(eq, ((-1,),), ((0, 0),))
    else:
        result = compile_gaussian_polyhedral_path(eq, ((-1,),), ((0, 0),))
        assert result is not None and result.certificate["meta"]["transcend_backend"] == "not_used"
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used() is sticky


def test_no_optional_backend_or_exponentials_are_needed_outside_strict_sealing(monkeypatch):
    eq = _equilibrium()

    def fail(*args):
        pytest.fail("exact path tried numerical exponential evaluation")

    monkeypatch.setattr(transcend, "exp_iv", fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    result = compile_gaussian_polyhedral_path(eq, ((-1,),), ((0, 0),))
    assert result is not None
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        compile_gaussian_polyhedral_path(eq, ((-1,),), ((0, 0),))


def test_factory_records_snapshot_mutable_inputs_and_integrity_evidence():
    with pytest.raises(TypeError):
        GaussianPolyhedralPath()
    eq = _equilibrium()
    target, beta = [[-1]], [[0, 0]]
    result = compile_gaussian_polyhedral_path(eq, target, beta)
    assert result is not None
    target[0][0] = beta[0][0] = 99
    assert result.target_coefficients == ((0,), (-1,)) and result.bias_coefficients[1] == (0, 0)
    with pytest.raises(FrozenInstanceError):
        result.status = "invalid"
    with pytest.raises(FrozenInstanceError):
        result.sign_tests[0].verdict = "negative"
    assert schema_errors_v1(result.certificate) == [] and certificate_is_unconditional(result.certificate)
    assert verify_certificate_digest(result.certificate)
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    saved = deepcopy(result.certificate)
    eq.certificate["payload"]["damaged"] = object()
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    result.certificate["payload"]["status"] = "invalid"
    assert not verify_certificate_digest(result.certificate)
