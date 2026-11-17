# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent joint target/bias derivatives and cached-proof contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    seal_certificate,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_entropy_bias import (
    GaussianEntropyBias,
    certify_gaussian_entropy_bias,
)
from omnibias.core.verified.gaussian_entropy_face import certify_gaussian_entropy_face
from omnibias.core.verified.gaussian_entropy_residual import certify_gaussian_entropy_residual
from omnibias.core.verified.gaussian_entropy_sensitivity import certify_gaussian_entropy_sensitivity
from omnibias.core.verified.gaussian_entropy_solver import solve_gaussian_entropy
from omnibias.core.verified.gaussian_entropy_unique import certify_gaussian_entropy_unique
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _source(*, centers=((-1,), (1,)), biases=(0, 0), viscosity=1, time=1, target=None,
            rows=(), bounds=(), reference=None, solve=False, budget=None, terms=16):
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=time, budget=budget)
    projector = compile_polyhedral_projector(model.dimension, rows, bounds, budget=budget)
    assert projector is not None
    target = (0,)*model.dimension if target is None else target
    domain = certify_gaussian_posterior_domain(model, projector, target)
    assert domain is not None
    unique = certify_gaussian_entropy_unique(domain, reference=(0,)*model.dimension if reference is None else reference)
    assert unique is not None
    if solve:
        solution = solve_gaussian_entropy(unique, tolerance=1e-12, max_iterations=32)
        assert solution is not None
        residual = solution.residual
    else:
        residual = certify_gaussian_entropy_residual(unique, (F(1, model.mode_count),)*model.mode_count)
        assert residual is not None
    face = certify_gaussian_entropy_face(residual)
    assert face is not None
    source = certify_gaussian_entropy_sensitivity(face, terms=terms)
    assert source is not None
    return source


def _bias(source=None, **kwargs):
    result = certify_gaussian_entropy_bias(_source(**kwargs) if source is None else source)
    assert result is not None
    return result


def _contains(interval, value):
    return F(interval.lo) <= value <= F(interval.hi)


def _vector_contains(vector, expected):
    assert len(vector) == len(expected)
    assert all(_contains(interval, exact) for interval, exact in zip(vector, expected, strict=True))


def _matrix_contains(matrix, expected):
    assert len(matrix) == len(expected)
    for row, exact in zip(matrix, expected, strict=True):
        _vector_contains(row, exact)


def _zero(vector):
    return all(interval.lo == interval.hi == 0 for interval in vector)


@pytest.mark.parametrize("terms", [1, 4, 16])
def test_scalar_closed_form_bias_response_and_joint_action(terms):
    result = _bias(terms=terms)
    assert result.projected_centers == ((F(0),), (F(2),))
    _vector_contains(result.projected_mean, (F(1),))
    _matrix_contains(result.bias_forcing, ((F(-1, 2), F(1, 2)),))
    _matrix_contains(result.bias_jacobian, ((F(-1), F(1)),))
    _vector_contains(result.jvp((F(3, 7),), (F(-2, 5), F(1, 5))), (F(51, 35),))
    target, bias = result.vjp((F(3, 7),))
    _vector_contains(target, (F(6, 7),))
    _vector_contains(bias, (F(-3, 7), F(3, 7)))


def test_nonunit_age_forcing_and_response_have_distinct_scalings():
    result = _bias(time=2)
    _matrix_contains(result.bias_forcing, ((F(-1, 4), F(1, 4)),))
    _matrix_contains(result.bias_jacobian, ((F(-2, 3), F(2, 3)),))
    _vector_contains(result.jvp((1,), (0, 1)), (F(10, 3),))
    target, bias = result.vjp((1,))
    _vector_contains(target, (F(8, 3),))
    _vector_contains(bias, (F(-2, 3), F(2, 3)))


def test_coupled_response_and_transpose_match_independent_exact_inverse():
    result = _bias(centers=((-1, -1), (1, 1)), viscosity=2)
    _matrix_contains(result.bias_jacobian, ((F(-1), F(1)), (F(-1), F(1))))
    _vector_contains(result.jvp((2, -1), (F(1, 3), F(4, 3))), (F(7, 2), F(1, 2)))
    target, bias = result.vjp((2, -1))
    _vector_contains(target, (F(5, 2), F(-1, 2)))
    _vector_contains(bias, (F(-1), F(1)))


def test_redundant_equality_rows_project_bias_response_and_cotangents():
    result = _bias(centers=((-1, 0), (1, 0)),
                   rows=((1, 1), (-1, -1), (2, 2)), bounds=(0, 0, 0))
    assert result.projected_centers == ((F(0), F(0)), (F(1), F(-1)))
    _matrix_contains(result.bias_jacobian, ((F(-1, 3), F(1, 3)), (F(1, 3), F(-1, 3))))
    _vector_contains(result.jvp((1, 0), (0, 1)), (F(1), F(-1)))
    target, bias = result.vjp((1, -1))
    _vector_contains(target, (F(4, 3), F(-4, 3)))
    _vector_contains(bias, (F(-2, 3), F(2, 3)))
    normal_target, normal_bias = result.vjp((2**2500, 2**2500))
    assert _zero(normal_target) and _zero(normal_bias)


def test_strict_active_normal_target_and_upstream_do_not_leak_into_bias():
    result = _bias(centers=((-1, -1), (1, 1)), viscosity=2,
                   rows=((0, 1),), bounds=(0,), target=(0, 1))
    _matrix_contains(result.bias_jacobian, ((F(-2, 3), F(2, 3)), (F(0), F(0))))
    assert result.jvp((1, 10**1000), (0, 1)) == result.jvp((1, 0), (0, 1))
    target, bias = result.vjp((0, 10**1000))
    assert _zero(target) and _zero(bias)


def test_exact_common_bias_gauge_is_removed_before_float_conversion():
    result = _bias()
    common = 2**2500
    assert _zero(result.jvp((0,), (common, common)))
    assert result.jvp((F(2, 7),), (common, common)) == result.jvp((F(2, 7),), (0, 0))
    assert result.jvp((0,), (common, common+1)) == result.jvp((0,), (0, 1))


def test_near_common_bias_direction_keeps_small_nonzero_response():
    result = _bias()
    delta = F(1, 2**100)
    response = result.jvp((0,), (1, 1+delta))
    _vector_contains(response, (delta,))
    assert response[0].lo > 0


def test_near_normal_cotangent_keeps_small_nonzero_bias_vjp():
    result = _bias(centers=((-1, 0), (1, 0)), rows=((1, 1), (-1, -1)), bounds=(0, 0))
    delta = F(1, 2**100)
    target, bias = result.vjp((1, 1+delta))
    _vector_contains(target, (-F(2, 3)*delta, F(2, 3)*delta))
    _vector_contains(bias, (F(1, 3)*delta, -F(1, 3)*delta))
    assert bias[0].lo > 0 and bias[1].hi < 0


@pytest.mark.parametrize("kind", ["single", "normal_modes", "singleton"])
def test_structurally_bias_independent_decisions_have_exact_zero_bias_response(kind):
    if kind == "single":
        result = _bias(centers=((F(1, 3), F(-1, 5)),), biases=(F(2, 7),), time=2,
                       rows=((1, 1), (-1, -1)), bounds=(0, 0))
        bias_direction = (17,)
    elif kind == "normal_modes":
        result = _bias(centers=((0, -1), (0, 1)), rows=((0, 1),), bounds=(0,), target=(0, 1))
        bias_direction = (-3, 19)
    else:
        result = _bias(centers=((-1, -1), (1, 1)), viscosity=2,
                       rows=((1, 0), (-1, 0), (0, 1), (0, -1)), bounds=(0, 0, 0, 0))
        bias_direction = (-3, 19)
    assert all(_zero(row) for row in result.bias_forcing)
    assert all(_zero(row) for row in result.bias_jacobian)
    assert _zero(result.jvp((0, 0), bias_direction))
    assert _zero(result.vjp((17, -23))[1])


def test_nonconvex_q_four_bias_response_matches_independent_high_precision_root():
    mp = pytest.importorskip("mpmath")
    result = _bias(centers=((-2,), (2,)), biases=(0, -12), viscosity=F(1, 2),
                   reference=(-2,), solve=True, terms=4)
    assert result.sensitivity.face.residual.unique.domain.model.contraction_bound == 4
    with mp.workdps(100):
        root = mp.findroot(lambda x: x-2*mp.tanh(2*x-6), (-2, mp.mpf("-1.9")))
        p = 1/(1+mp.exp(12-4*root))
        response = 4*p*(1-p)/(1-16*p*(1-p))
        lower, upper = result.bias_jacobian[0]
        assert mp.mpf(lower.lo) <= -response <= mp.mpf(lower.hi)
        assert mp.mpf(upper.lo) <= response <= mp.mpf(upper.hi)


def test_nonuniform_three_mode_joint_response_matches_original_gaussian_oracle():
    mp = pytest.importorskip("mpmath")
    centers = ((-1, 0), (1, 1), (0, -1))
    biases = (F(0), F(1, 3), F(-1, 5))
    target = (F(1, 7), F(-1, 9))
    result = _bias(centers=centers, biases=biases, viscosity=2, time=3, target=target, solve=True)
    xi, beta, cotangent = (F(2, 7), F(-3, 11)), (F(1, 3), F(-2, 5), F(4, 7)), (F(5, 13), F(-2, 9))
    with mp.workdps(100):
        def exact(value):
            rational = F(value)
            return mp.mpf(rational.numerator)/rational.denominator

        modes = tuple(mp.matrix(row) for row in centers)
        v = mp.matrix(tuple(map(exact, target)))

        def probabilities(point):
            unnormalized = tuple(mp.exp(exact(bias)-mp.fdot(point-mode, point-mode)/24)
                                 for mode, bias in zip(modes, biases, strict=True))
            total = mp.fsum(unnormalized)
            return tuple(weight/total for weight in unnormalized)

        def stationarity(x, y):
            point = mp.matrix((x, y))
            p = probabilities(point)
            mean = sum((weight*mode for weight, mode in zip(p, modes, strict=True)), mp.zeros(2, 1))
            residual = point-3*v-mean
            return residual[0], residual[1]

        point = mp.matrix(mp.findroot(stationarity, (3*v[0], 3*v[1])))
        posterior = probabilities(point)
        mean = sum((p*mode for p, mode in zip(posterior, modes, strict=True)), mp.zeros(2, 1))
        covariance = sum((p*(mode-mean)*(mode-mean).T for p, mode in zip(posterior, modes, strict=True)), mp.zeros(2))
        inverse = (mp.eye(2)-covariance/12)**-1
        jacobian = mp.matrix(2, 3)
        for j, (p, mode) in enumerate(zip(posterior, modes, strict=True)):
            response = inverse*(p*(mode-mean))
            for i in range(2):
                jacobian[i, j] = response[i]
                interval = result.bias_jacobian[i][j]
                assert mp.mpf(interval.lo) <= response[i] <= mp.mpf(interval.hi)
        joint = 3*inverse*mp.matrix(tuple(map(exact, xi)))+jacobian*mp.matrix(tuple(map(exact, beta)))
        for interval, reference in zip(result.jvp(xi, beta), joint, strict=True):
            assert mp.mpf(interval.lo) <= reference <= mp.mpf(interval.hi)
        target_vjp, bias_vjp = result.vjp(cotangent)
        g = mp.matrix(tuple(map(exact, cotangent)))
        for intervals, reference in ((target_vjp, 3*inverse*g), (bias_vjp, jacobian.T*g)):
            for interval, value in zip(intervals, reference, strict=True):
                assert mp.mpf(interval.lo) <= value <= mp.mpf(interval.hi)


def test_underflowed_posterior_keeps_nonzero_bias_bounds_with_sufficient_exact_bits():
    mp = pytest.importorskip("mpmath")
    result = _bias(centers=((0,), (10,)), biases=(0, -1000), reference=(0,), solve=True, terms=4,
                   budget=RidgeBudget(max_coefficient_bits=8192))
    assert result.sensitivity.contraction_bound > 0
    assert result.bias_jacobian[0][0].lo < 0 < result.bias_jacobian[0][1].hi
    with mp.workdps(600):
        root = mp.findroot(lambda x: x-10/(1+mp.exp(1025-5*x)), (0, mp.exp(-1020)))
        p = 1/(1+mp.exp(1025-5*root))
        derivative = 10*p*(1-p)/(1-50*p*(1-p))
        assert 0 < derivative < mp.mpf(2)**-1074
        negative, positive = result.bias_jacobian[0]
        assert mp.mpf(negative.lo) <= -derivative <= mp.mpf(negative.hi)
        assert mp.mpf(positive.lo) <= derivative <= mp.mpf(positive.hi)
    assert _zero(result.jvp((0,), (2**2500, 2**2500)))


def test_factory_record_and_nested_certificate_snapshot_are_immutable():
    result = _bias()
    assert verify_certificate_digest(result.certificate)
    assert certificate_is_unconditional(result.certificate)
    assert schema_errors_v1(result.certificate) == []
    with pytest.raises(TypeError):
        GaussianEntropyBias()
    with pytest.raises(FrozenInstanceError):
        result.projected_centers = ()
    saved = deepcopy(result.certificate)
    result.sensitivity.certificate["payload"]["type"] = "damaged"
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    assert certify_gaussian_entropy_bias(result.sensitivity) is None
    with pytest.raises(ValueError):
        result.jvp((0,), (0, 0))
    with pytest.raises(ValueError):
        result.vjp((0,))


@pytest.mark.parametrize("strict,sticky", [(False, False), (False, True), (True, False), (True, True)])
def test_cached_joint_queries_need_no_backend_solve_or_new_certificate(strict, sticky, monkeypatch):
    result = _bias(centers=((-1, -1), (1, 1)), viscosity=2)

    def fail(*args, **kwargs):
        pytest.fail("cached bias query used a solver, transcendental or new proof seal")

    for module_name in ("polyhedral_projection", "gaussian_entropy_residual", "gaussian_projection_minimum"):
        module = import_module("omnibias.core.verified."+module_name)
        monkeypatch.setattr(module, "_project_exact", fail)
    linear = import_module("omnibias.core._rational_linear")
    monkeypatch.setattr(linear, "solve_affine", fail)
    monkeypatch.setattr(linear, "factor_positive_definite", fail)
    proof = import_module("omnibias.core.proof.certificate")
    monkeypatch.setattr(proof, "make_certificate", fail)
    for name in ("ln_iv", "exp_iv", "tanh_iv"):
        monkeypatch.setattr(transcend, name, fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    _vector_contains(result.jvp((2, -1), (0, 1)), (F(7, 2), F(1, 2)))
    target, bias = result.vjp((2, -1))
    _vector_contains(target, (F(5, 2), F(-1, 2)))
    _vector_contains(bias, (F(-1), F(1)))
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used() is sticky


@pytest.mark.parametrize("strict", [False, True])
def test_factory_requires_fresh_untainted_backend_and_preserves_state(strict, monkeypatch):
    source = _source()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    _bias(source)
    assert transcend.strict_backend() is strict
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        certify_gaussian_entropy_bias(source)
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used()


def test_factory_requires_rigorous_backend_even_with_cached_sensitivity(monkeypatch):
    source = _source()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_entropy_bias(source)


@pytest.mark.parametrize("dependency", [False, True])
def test_conditional_proof_cannot_be_promoted_or_queried(dependency):
    result = _bias()
    changed = result.sensitivity.certificate if dependency else result.certificate
    changed["meta"]["transcend_backend"] = "libm"
    sealed = seal_certificate(changed)
    changed.clear()
    changed.update(sealed)
    if dependency:
        assert certify_gaussian_entropy_bias(result.sensitivity) is None
    with pytest.raises(ValueError):
        result.jvp((0,), (0, 0))
    with pytest.raises(ValueError):
        result.vjp((0,))


def test_factory_rejects_invalid_source_type():
    with pytest.raises(TypeError):
        certify_gaussian_entropy_bias(object())


@pytest.mark.parametrize("target,bias,error", [((True,), (0, 0), TypeError), ((0.0,), (0, 0), TypeError),
    ((0,), (0, True), TypeError), ((0,), (0, F(1, 2), 1), ValueError),
    ((), (0, 0), ValueError), ((0, 0), (0, 0), ValueError), ((0,), (0, "1"), TypeError)])
def test_joint_input_validation_precedes_resource_reservation(target, bias, error, monkeypatch):
    result = _bias()
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_a: pytest.fail("invalid query reached preflight"))
    with pytest.raises(error):
        result.jvp(target, bias)


@pytest.mark.parametrize("cotangent,error", [((False,), TypeError), ((1.0,), TypeError), ((), ValueError), ((0, 0), ValueError)])
def test_cotangent_validation_precedes_resource_reservation(cotangent, error, monkeypatch):
    result = _bias()
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_a: pytest.fail("invalid query reached preflight"))
    with pytest.raises(error):
        result.vjp(cotangent)


def test_exact_query_bit_budget_refuses_even_a_common_oversized_gauge():
    result = _bias()
    huge = 10**1500
    with pytest.raises(ValueError):
        result.jvp((0,), (huge, huge))
    with pytest.raises(ValueError):
        result.vjp((huge,))


@pytest.mark.parametrize("field", ["projected_centers", "projected_mean", "bias_forcing", "bias_jacobian"])
def test_bound_runtime_field_tamper_refuses_zero_queries(field):
    result = _bias()
    if field == "projected_centers":
        changed = ((F(0),), (F(3),))
    elif field == "projected_mean":
        changed = (Interval.point(0.0),)
    else:
        changed = ((Interval.point(0.0), Interval.point(0.0)),)
    object.__setattr__(result, field, changed)
    assert verify_certificate_digest(result.certificate)
    with pytest.raises(ValueError):
        result.jvp((0,), (0, 0))
    with pytest.raises(ValueError):
        result.vjp((0,))


@pytest.mark.parametrize("field", ["polynomial", "projected_covariance", "tail_norm_bound"])
def test_sensitivity_field_tamper_is_not_hidden_by_its_valid_digest(field):
    result = _bias()
    changed = F(0) if field == "tail_norm_bound" else ((Interval.point(0.0),),)
    object.__setattr__(result.sensitivity, field, changed)
    assert verify_certificate_digest(result.sensitivity.certificate)
    assert certify_gaussian_entropy_bias(result.sensitivity) is None
    with pytest.raises(ValueError):
        result.jvp((0,), (0, 0))


@pytest.mark.parametrize("level", ["face", "residual", "unique", "domain"])
def test_deep_source_damage_refuses_even_algebraically_zero_queries(level):
    result = _bias()
    face = result.sensitivity.face
    dependencies = {"face": face, "residual": face.residual, "unique": face.residual.unique,
                    "domain": face.residual.unique.domain}
    dependencies[level].certificate["payload"]["type"] = "damaged"
    assert certify_gaussian_entropy_bias(result.sensitivity) is None
    with pytest.raises(ValueError):
        result.vjp((0,))


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_factory_resource_preflight_precedes_nested_proof_scan(guard, monkeypatch):
    source = _source()
    module = import_module("omnibias.core.verified.gaussian_entropy_bias")

    def exhausted(*args):
        raise ValueError("test bias resource exhaustion")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(module, "_sensitivity_intact", lambda *_a: pytest.fail("unreserved sensitivity scan"))
    assert certify_gaussian_entropy_bias(source) is None


@pytest.mark.parametrize("guard", ["_entries", "_work"])
@pytest.mark.parametrize("query", ["jvp", "vjp"])
def test_query_resource_preflight_precedes_cached_proof_scan(guard, query, monkeypatch):
    result = _bias()
    module = import_module("omnibias.core.verified.gaussian_entropy_bias")

    def exhausted(*args):
        raise ValueError("test bias resource exhaustion")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(module, "_bias_intact", lambda *_a: pytest.fail("unreserved bias scan"))
    with pytest.raises(ValueError, match="resource"):
        result.jvp((0,), (0, 0)) if query == "jvp" else result.vjp((0,))


def test_bias_response_can_exhaust_bits_after_valid_source_without_partial_result():
    source = _source(terms=8, budget=RidgeBudget(max_coefficient_bits=128))
    assert source.tail_norm_bound > 0 and verify_certificate_digest(source.certificate)
    assert certify_gaussian_entropy_bias(source) is None


def test_factory_resource_bound_covers_actual_shared_meter_and_queries(monkeypatch):
    source = _source(centers=((-1, 0), (1, 1), (0, -1)), biases=(0, F(1, 3), F(-1, 5)),
                     viscosity=2, time=3, target=(F(1, 7), F(-1, 9)), solve=True)
    original = _Arithmetic._work
    charges = []

    def observed(self, work):
        charges.append(work)
        return original(self, work)

    monkeypatch.setattr(_Arithmetic, "_work", observed)
    result = _bias(source)
    assert charges and all(charge >= 0 for charge in charges)
    factory_work = sum(charges)
    assert factory_work < source.face.residual.unique.domain.model.budget.max_elimination_work
    charges.clear()
    result.jvp((1, -1), (F(1, 3), F(2, 3), F(-1, 3)))
    assert 0 < sum(charges) < factory_work
    charges.clear()
    result.vjp((1, -1))
    assert 0 < sum(charges) < factory_work
