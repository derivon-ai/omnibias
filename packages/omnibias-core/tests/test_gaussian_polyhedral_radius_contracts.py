# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Quantitative Gaussian path certificates preserve finite proof contracts."""

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
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_polyhedral_equilibrium import (
    certify_gaussian_polyhedral_equilibrium,
)
from omnibias.core.verified.gaussian_polyhedral_path import compile_gaussian_polyhedral_path
from omnibias.core.verified.gaussian_polyhedral_radius import (
    GaussianPolyhedralPathRadius,
    certify_gaussian_polyhedral_path_radius,
)


def _path(*, selected=(), order=7, shift=0, gauge=0, budget=None, constant=False):
    model = GaussianHeatMap(((shift-1,), (shift+1,)), (F(1, 4),)*2, budget=budget)
    base = restrict_gaussian_heat(model, (), ())
    assert base is not None
    face = compile_gaussian_polyhedral_face(base, ((1,),), (shift,), selected)
    assert face is not None
    eq = certify_gaussian_polyhedral_equilibrium(face, (0,), (shift,))
    assert eq is not None
    targets = ((0 if constant else -1,),)+((0,),)*(order-1)
    biases = ((gauge, gauge+(0 if constant else 2)),)+((gauge, gauge),)*(order-1)
    path = compile_gaussian_polyhedral_path(eq, targets, biases)
    assert path is not None
    return path


def _radius(path, **kwargs):
    options = {"parameter_radius": F(1, 20), "spatial_radius": F(1, 4), "radius": F(1, 125)}
    options.update(kwargs)
    return certify_gaussian_polyhedral_path_radius(path, **options)


@pytest.mark.parametrize("selected", [(), (0,)])
def test_rational_bounds_and_parent_binding(selected):
    path = _path(selected=selected)
    result = _radius(path)
    assert result is not None and result.path is path and result.analytic.series is path.series
    assert result.status == path.status
    assert result.sign_bounds[0].sign is path.sign_tests[0]
    bound = result.sign_bounds[0]
    assert bound.certified and bound.margin > 0
    assert bound.margin == abs(bound.sign.leading_coefficient)-bound.polynomial_remainder_bound-bound.scaled_tail_bound
    ratio = result.radius/result.analytic.parameter_radius
    expected = result.analytic.spatial_radius*ratio**8/(1-ratio)
    assert result.tail_bound(result.radius) == expected
    assert result.tail_bound(0) == 0
    assert result.certificate["payload"]["sign_interval"] == "0<t<=radius"
    assert result.enclose(0)[0].lo <= 0 <= result.enclose(0)[0].hi


@pytest.mark.parametrize("selected", [(), (0,)])
def test_spatial_translation_and_common_bias_polynomial_leave_sign_bounds_unchanged(selected):
    baseline = _radius(_path(selected=selected))
    shifted = _radius(_path(selected=selected, shift=2**1800, gauge=2**2500))
    assert baseline is not None and shifted is not None
    for name in ("status", "sign_bounds", "radius", "eventual_active_indices"):
        assert getattr(baseline, name) == getattr(shifted, name)
    for name in ("phase_half_width_bound", "complex_contraction_bound", "target_growth_bound", "bias_growth_bound"):
        assert getattr(baseline.analytic, name) == getattr(shifted.analytic, name)
    # The proof remains exact at these coordinates; binary64 rendering is
    # a separate boundary and cannot return finite representable endpoints.
    with pytest.raises((ArithmeticError, ValueError, OverflowError)):
        shifted.enclose(0)


@pytest.mark.parametrize("selected", [(), (0,)])
def test_structural_zero_requires_no_positive_margin(selected):
    result = _radius(_path(selected=selected, constant=True))
    assert result is not None and result.status == "valid"
    assert result.eventual_active_indices == (0,)
    sign = result.sign_bounds[0]
    assert sign.sign.verdict == "zero" and sign.sign.identity is not None
    assert sign.certified and sign.change_bound == sign.margin == sign.scaled_tail_bound == sign.polynomial_remainder_bound == 0


def test_inconclusive_prefix_cannot_be_promoted_by_choosing_an_arbitrarily_small_radius(monkeypatch):
    path = _path(order=2)
    assert path.status == "inconclusive"
    module = import_module("omnibias.core.verified.gaussian_polyhedral_radius")

    def fail(*args):
        pytest.fail("inconclusive path reached analytic certification")

    monkeypatch.setattr(module, "_certify_bias_series", fail)
    assert _radius(path, radius=F(1, 2**2000)) is None


@pytest.mark.parametrize("name,bad,error", [
    ("parameter_radius", 0, ValueError), ("parameter_radius", -1, ValueError),
    ("spatial_radius", 0, ValueError), ("spatial_radius", -1, ValueError),
    ("radius", 0, ValueError), ("radius", -1, ValueError),
    ("radius", F(1, 20), ValueError), ("radius", F(1, 10), ValueError),
    ("parameter_radius", 0.05, TypeError), ("spatial_radius", True, TypeError),
    ("radius", F(1, 125)+0j, TypeError), ("radius", None, TypeError),
])
def test_invalid_radius_metadata_raises_before_preflight(name, bad, error, monkeypatch):
    path = _path()

    def fail(*args):
        pytest.fail("invalid scalar reached resource preflight")

    monkeypatch.setattr(_Arithmetic, "_entries", fail)
    with pytest.raises(error):
        _radius(path, **{name: bad})


def test_invalid_path_type_and_direct_construction_raise():
    with pytest.raises(TypeError):
        _radius(None)
    with pytest.raises(TypeError):
        GaussianPolyhedralPathRadius()


@pytest.mark.parametrize("dependency", ["path", "equilibrium", "face", "base", "geometry", "restriction"])
def test_factory_checks_all_live_dependency_digests_before_exact_actions(dependency, monkeypatch):
    path = _path(selected=(0,))
    eq = path.equilibrium
    record = {"path": path, "equilibrium": eq, "face": eq.face, "base": eq.face.base_restriction,
              "geometry": eq.face.geometry, "restriction": eq.face.geometry.restriction}[dependency]
    record.certificate["payload"]["damaged"] = object()

    def fail(*args):
        pytest.fail("damaged proof reached exact work")

    monkeypatch.setattr(_Arithmetic, "exact", fail)
    assert _radius(path) is None


@pytest.mark.parametrize("dependency", ["radius", "analytic", "path", "equilibrium", "face", "base"])
def test_evaluation_checks_nested_evidence_without_resealing(dependency):
    result = _radius(_path())
    assert result is not None
    eq = result.path.equilibrium
    record = {"radius": result, "analytic": result.analytic, "path": result.path,
              "equilibrium": eq, "face": eq.face, "base": eq.face.base_restriction}[dependency]
    record.certificate["payload"]["damaged"] = object()
    for method in (result.enclose, result.tail_bound):
        with pytest.raises(ValueError, match="damaged"):
            method(F(1, 1000))


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_preflight_refuses_before_exact_actions(guard, monkeypatch):
    path = _path()

    def exhausted(*args):
        raise ValueError("test exhausted budget")

    def fail(*args):
        pytest.fail("failed preflight reached exact data")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(_Arithmetic, "exact", fail)
    assert _radius(path) is None


def test_analytic_and_sign_work_use_one_cumulative_meter(monkeypatch):
    path = _path()
    observed, child_work = [], []
    module = import_module("omnibias.core.verified.gaussian_polyhedral_radius")
    original = module._certify_bias_series

    def count(self, work):
        observed.append(work)
        assert work <= self.max_elimination_work

    def child(*args):
        arithmetic = args[-1]
        child_work.append(arithmetic.used)
        result = original(*args)
        child_work.append(arithmetic.used)
        return result

    monkeypatch.setattr(RidgeBudget, "_work", count)
    monkeypatch.setattr(module, "_certify_bias_series", child)
    result = _radius(path)
    assert result is not None
    assert observed[0] < child_work[0] < child_work[1] < observed[-1]
    assert all(a < b for a, b in zip(observed, observed[1:], strict=False))
    observed.clear()
    result.enclose(F(1, 1000))
    assert all(a < b for a, b in zip(observed, observed[1:], strict=False))


def test_mid_certificate_work_and_bit_exhaustion_refuse(monkeypatch):
    path = _path()
    seen = []

    def bounded(self, work):
        seen.append(work)
        if work > seen[0]+50:
            raise ValueError("aggregate budget exhausted")

    with monkeypatch.context() as scope:
        scope.setattr(RidgeBudget, "_work", bounded)
        assert _radius(path) is None
    assert seen[-1] > seen[0]+50
    assert _radius(path, radius=F(1, 2**5000)) is None


@pytest.mark.parametrize("method_name", ["tail_bound", "enclose"])
@pytest.mark.parametrize("parameter,error", [(-1, ValueError), (F(1, 124), ValueError), (0.0, TypeError), (False, TypeError)])
def test_queries_enforce_the_certified_closed_real_interval(method_name, parameter, error):
    result = _radius(_path())
    assert result is not None
    with pytest.raises(error):
        getattr(result, method_name)(parameter)


def test_no_new_series_factorization_exponentials_or_backend_are_needed(monkeypatch):
    path = _path()

    def fail(*args, **kwargs):
        pytest.fail("radius construction/evaluation attempted a new solve, exp or series")

    monkeypatch.setattr(RationalLDLT, "solve", fail)
    for module_name, names in (
        ("omnibias.core._rational_linear", ("factor_positive_definite",)),
        ("omnibias.core.gaussian_bias_jet", ("_build_equal_logits", "gaussian_bias_inverse_series")),
        ("omnibias.core.verified.transcend", ("exp_iv",)),
    ):
        module = import_module(module_name)
        for name in names:
            monkeypatch.setattr(module, name, fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    result = _radius(path)
    assert result is not None and result.certificate["meta"]["transcend_backend"] == "not_used"
    result.enclose(F(1, 1000))
    assert transcend.libm_fallback_used() and not transcend.strict_backend()


@pytest.mark.parametrize("sticky", [False, True])
def test_strict_sealing_preserves_policy_and_fallback_history(sticky, monkeypatch):
    path = _path()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    if sticky:
        with pytest.raises(RuntimeError, match="libm_fallback"):
            _radius(path)
    else:
        assert _radius(path) is not None
    assert transcend.strict_backend() and transcend.libm_fallback_used() is sticky


def test_strict_mode_cannot_seal_without_optional_backend(monkeypatch):
    path = _path()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        _radius(path)


def test_record_freezing_snapshot_and_unconditional_certificate_schema():
    path = _path()
    result = _radius(path)
    assert result is not None
    with pytest.raises(FrozenInstanceError):
        result.radius = F(1)
    with pytest.raises(FrozenInstanceError):
        result.sign_bounds[0].certified = False
    assert schema_errors_v1(result.certificate) == []
    assert certificate_is_unconditional(result.certificate)
    assert verify_certificate_digest(result.certificate)
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    saved = deepcopy(result.certificate)
    path.certificate["payload"]["damaged"] = object()
    result.analytic.certificate["payload"]["damaged"] = object()
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
