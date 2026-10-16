# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Original KKT multipliers, singleton posteriors and proof composition."""

import math
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_affine_adjoint import compile_gaussian_affine_adjoint
from omnibias.core.verified.gaussian_affine_stationarity import (
    GaussianAffineStationarity,
    certify_gaussian_affine_stationarity,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import certify_gaussian_affine_inverse
from omnibias.core.verified.mean_map import MeanMap


def _compile(source, matrix, rhs):
    restriction = restrict_gaussian_heat(source, matrix, rhs)
    assert restriction is not None
    geometry = compile_gaussian_affine_adjoint(restriction)
    assert geometry is not None
    return geometry


def _axis(**kwargs):
    source = GaussianHeatMap(((-1, -2), (1, 2)), (F(9, 4), F(1, 4)), **kwargs)
    return _compile(source, ((0, 1),), (1,))


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _original(mp, model, point):
    nu, age = _mp(mp, model.viscosity), _mp(mp, model.time)
    centers = [tuple(_mp(mp, value) for value in center) for center in model.centers]
    weights = [mp.exp(_mp(mp, bias)-sum((x-a)**2 for x, a in zip(point, center, strict=True))/(4*nu*age))
               for center, bias in zip(centers, model.biases, strict=True)]
    probabilities = tuple(value/sum(weights) for value in weights)
    mean = tuple(sum(p*center[i] for p, center in zip(probabilities, centers, strict=True))
                 for i in range(model.dimension))
    velocity = tuple((x-a)/age for x, a in zip(point, mean, strict=True))
    return probabilities, velocity


def _contains(mp, interval, exact):
    assert _mp(mp, interval.lo) <= exact <= _mp(mp, interval.hi)


@pytest.mark.parametrize("normal_target", [0, 7, F(-1, 3)])
def test_original_multiplier_has_correct_sign_and_normal_target_dependence(normal_target):
    geometry = _axis()
    result = certify_gaussian_affine_stationarity(geometry, (0, normal_target), (0, 1))
    assert result is not None
    assert geometry.restriction.source_model.contraction_bound == F(5, 2)
    assert geometry.restriction.model.contraction_bound == F(1, 2)
    assert result.root_enclosure[0].contains(0) and result.root_enclosure[1].contains(1)
    for probability in result.posterior:
        assert probability.contains(F(1, 2))
    multiplier = result.primal_multiplier[0]
    assert F(multiplier.lo) <= normal_target-1 <= F(multiplier.hi)
    assert multiplier.hi-multiplier.lo < 1e-11
    # Original U(0,1)=(0,1), so U-target+C^T lambda=0 gives lambda=target_y-1.
    assert result.root_certificate.target == result.target == (0, normal_target)


def test_oblique_original_gaussian_kkt_root_and_multiplier_against_high_precision():
    mp = pytest.importorskip("mpmath")
    source = GaussianHeatMap(((0, 0), (1, 2), (3, -1)), (F(1, 3), F(-1, 7), F(2, 5)),
                             viscosity=F(3, 2), time=4)
    geometry = _compile(source, ((1, 2),), (3,))
    target = (F(1, 3), F(-1, 5))
    with mp.workdps(90):
        goal = tuple(_mp(mp, value) for value in target)

        def point(parameter):
            return (_mp(mp, F(3, 5))+2*parameter, _mp(mp, F(6, 5))-parameter)

        def tangential_residual(parameter):
            _, velocity = _original(mp, source, point(parameter))
            return 2*(velocity[0]-goal[0])-(velocity[1]-goal[1])

        parameter = mp.findroot(tangential_residual, 0, tol=mp.mpf("1e-80"))
        exact_point = point(parameter)
        posterior, velocity = _original(mp, source, exact_point)
        multiplier = ((goal[0]-velocity[0])+2*(goal[1]-velocity[1]))/5
        result = certify_gaussian_affine_stationarity(geometry, target, tuple(float(x/4) for x in exact_point))
        assert result is not None
        for interval, exact in zip(result.root_enclosure, exact_point, strict=True):
            _contains(mp, interval, exact)
        for interval, exact in zip(result.posterior, posterior, strict=True):
            _contains(mp, interval, exact)
        _contains(mp, result.primal_multiplier[0], multiplier)
        assert result.primal_multiplier[0].hi-result.primal_multiplier[0].lo < 1e-10


def test_full_rank_singleton_still_evaluates_nonconstant_original_posterior():
    mp = pytest.importorskip("mpmath")
    source = GaussianHeatMap(((0, 0), (1, 2)), (0, F(1, 3)), viscosity=F(3, 2), time=2)
    fixed = (F(1, 2), F(-3, 4))
    geometry = _compile(source, ((1, 0), (0, 1)), fixed)
    target = (F(2, 3), F(-1, 5))
    result = certify_gaussian_affine_stationarity(geometry, target, (1e200, -1e200))
    assert result is not None and result.root_certificate.root_certificate is None
    assert result.posterior and result.root_certificate.error_bound == 0
    with mp.workdps(90):
        posterior, velocity = _original(mp, source, tuple(_mp(mp, value) for value in fixed))
        for interval, exact in zip(result.posterior, posterior, strict=True):
            _contains(mp, interval, exact)
        for interval, v, u in zip(result.primal_multiplier, target, velocity, strict=True):
            _contains(mp, interval, _mp(mp, v)-u)
    assert not result.primal_multiplier[0].contains(0)


def test_affine_tangent_map_requires_original_normal_mean():
    mp = pytest.importorskip("mpmath")
    source = GaussianHeatMap(((2, -1), (2, 3)), (0, F(1, 3)), time=2)
    geometry = _compile(source, ((0, 1),), (1,))
    result = certify_gaussian_affine_stationarity(geometry, (F(1, 4), F(-1, 5)), (99, -99))
    assert result is not None and result.root_certificate.root_certificate is None
    assert result.root_certificate.feasible_point == (F(5, 2), F(1))
    with mp.workdps(90):
        _, velocity = _original(mp, source, (mp.mpf("2.5"), mp.mpf(1)))
        _contains(mp, result.primal_multiplier[0], -mp.mpf(1)/5-velocity[1])
    assert result.posterior


def test_coarse_nonfeasible_proposal_encloses_the_true_multiplier():
    result = certify_gaussian_affine_stationarity(_axis(), (0, 0), (3, -20))
    assert result is not None
    assert result.root_certificate.error_bound > 1
    assert result.primal_multiplier[0].contains(-1)
    assert result.root_enclosure[0].contains(0) and result.root_enclosure[1].contains(1)


def test_common_exact_bias_gauge_is_removed_before_binary_conversion():
    normal = _axis()
    source = normal.restriction.source_model
    shifted = _compile(GaussianHeatMap(source.centers, tuple(value+10**400 for value in source.biases)), ((0, 1),), (1,))
    a = certify_gaussian_affine_stationarity(normal, (0, 0), (0, 1))
    b = certify_gaussian_affine_stationarity(shifted, (0, 0), (0, 1))
    assert a is not None and b is not None
    assert a.posterior == b.posterior and a.primal_multiplier == b.primal_multiplier


@pytest.mark.parametrize("affine", [False, True])
def test_empty_constraints_have_no_multiplier_or_extra_posterior(affine):
    source = GaussianHeatMap(((2,),), (0,)) if affine else GaussianHeatMap(((-1,), (1,)), (0, 0), time=2)
    geometry = _compile(source, (), ())
    result = certify_gaussian_affine_stationarity(geometry, (0,), (0,))
    assert result is not None and result.primal_multiplier == result.posterior == ()
    assert result.root_certificate is not None
    assert result.root_enclosure[0].contains(2 if affine else 0)


def test_no_inverse_iteration_factorization_or_dense_covariance(monkeypatch):
    geometry = _axis()

    def fail(*args, **kwargs):
        pytest.fail("stationarity proof attempted an inverse iteration or factorization")

    for module_name, names in (
        ("omnibias.core.verified.inverse_gaussian_heat", ("inverse_gaussian_heat",)),
        ("omnibias.core.verified.inverse_gaussian_affine", ("inverse_gaussian_affine", "inverse_gaussian_heat")),
        ("omnibias.core._rational_linear", ("factor_positive_definite", "solve_affine")),
        ("omnibias.core.verified.gaussian_affine_adjoint", ("factor_positive_definite",)),
        ("omnibias.core.verified.linalg", ("_float_inverse", "_symmetric_ldlt_solve", "interval_solve")),
    ):
        module = import_module(module_name)
        for name in names:
            monkeypatch.setattr(module, name, fail)
    monkeypatch.setattr(MeanMap, "jacobian", fail)
    monkeypatch.setattr(GaussianHeatMap, "scaled_jacobian", fail)
    monkeypatch.setattr(GaussianHeatMap, "float_scaled_fixed_point", fail)
    assert certify_gaussian_affine_stationarity(geometry, (0, 0), (0, 1)) is not None


def test_factory_only_immutable_record_and_certificate_snapshots():
    with pytest.raises(TypeError):
        GaussianAffineStationarity()
    geometry = _axis()
    target, point = [0, 0], [0, 1]
    result = certify_gaussian_affine_stationarity(geometry, target, point)
    assert result is not None
    with pytest.raises(FrozenInstanceError):
        result.primal_multiplier = ()
    assert verify_certificate_digest(result.certificate)
    assert schema_errors_v1(result.certificate) == []
    assert certificate_is_unconditional(result.certificate)
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    target[0] = point[0] = 99
    assert result.target == (0, 0)
    saved = deepcopy(result.certificate)
    result.root_certificate.certificate["payload"]["damaged"] = object()
    geometry.certificate["payload"]["damaged"] = object()
    geometry.restriction.certificate["payload"]["damaged"] = object()
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    result.certificate["payload"]["target"][0] = [99, 1]
    assert not verify_certificate_digest(result.certificate)


@pytest.mark.parametrize("dependency", ["geometry", "restriction"])
def test_damaged_geometry_is_refused_before_composition(dependency, monkeypatch):
    geometry = _axis()
    damaged = geometry.certificate if dependency == "geometry" else geometry.restriction.certificate
    damaged["payload"]["damaged"] = object()

    def fail(*args, **kwargs):
        pytest.fail("damaged geometry must refuse before the root proof")

    module = import_module("omnibias.core.verified.gaussian_affine_stationarity")
    monkeypatch.setattr(module, "certify_gaussian_affine_inverse", fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    assert certify_gaussian_affine_stationarity(geometry, (0, 0), (0, 1)) is None


@pytest.mark.parametrize("dependency", ["affine", "numerical"])
def test_damaged_composed_root_is_refused(dependency, monkeypatch):
    geometry = _axis()
    root = certify_gaussian_affine_inverse(geometry.restriction, (0, 0), (0, 1))
    assert root is not None and root.root_certificate is not None
    damaged = root.certificate if dependency == "affine" else root.root_certificate.certificate
    damaged["payload"]["damaged"] = object()
    module = import_module("omnibias.core.verified.gaussian_affine_stationarity")
    monkeypatch.setattr(module, "certify_gaussian_affine_inverse", lambda *args, **kwargs: root)
    assert certify_gaussian_affine_stationarity(geometry, (0, 0), (0, 1)) is None


@pytest.mark.parametrize("sticky", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_rigorous_proof_preserves_prior_backend_state(sticky, strict, monkeypatch):
    geometry = _axis()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    result = certify_gaussian_affine_stationarity(geometry, (0, 0), (0, 1))
    assert result is not None and result.certificate["meta"]["transcend_backend"] == "mpmath"
    assert transcend.libm_fallback_used() is sticky and transcend.strict_backend() is strict


@pytest.mark.parametrize("full_rank", [False, True])
def test_missing_backend_raises_even_when_full_rank_root_is_exact(full_rank, monkeypatch):
    geometry = _axis()
    if full_rank:
        geometry = _compile(geometry.restriction.source_model, ((1, 0), (0, 1)), (0, 1))
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_affine_stationarity(geometry, (0, 0), (0, 1))
    assert not transcend.strict_backend()


def test_exact_empty_constraint_case_uses_no_backend_but_respects_strict_policy(monkeypatch):
    geometry = _compile(GaussianHeatMap(((2,),), (10**400,)), (), ())
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    result = certify_gaussian_affine_stationarity(geometry, (F(1, 3),), (-1e200,))
    assert result is not None and result.primal_multiplier == result.posterior == ()
    assert result.certificate["meta"]["transcend_backend"] == "not_used"
    assert transcend.libm_fallback_used() and not transcend.strict_backend()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_affine_stationarity(geometry, (0,), (0,))


@pytest.mark.parametrize("index", [1, 2])
@pytest.mark.parametrize("bad,error", [(None, TypeError), ((), ValueError), ((0,), ValueError),
    ((0, 0, 0), ValueError), ((True, 0), TypeError), ((complex(1), 0), TypeError)])
def test_vector_type_and_shape_validation(index, bad, error):
    arguments = [_axis(), (0, 0), (0, 1)]
    arguments[index] = bad
    with pytest.raises(error):
        certify_gaussian_affine_stationarity(*arguments)


@pytest.mark.parametrize("bad,error", [(F(1), TypeError), (math.inf, ValueError),
    (math.nan, ValueError), (2**53+1, ValueError)])
def test_binary64_scaled_point_contract(bad, error):
    with pytest.raises(error):
        certify_gaussian_affine_stationarity(_axis(), (0, 0), (bad, 1))


def test_invalid_geometry_and_exact_target_raise():
    with pytest.raises(TypeError):
        certify_gaussian_affine_stationarity(None, (0,), (0,))
    with pytest.raises(TypeError):
        certify_gaussian_affine_stationarity(_axis(), (0.0, 0), (0, 1))


@pytest.mark.parametrize("time", [F(1, 2), F(1, 4)])
def test_failed_contraction_refuses_before_root_evaluation(time, monkeypatch):
    geometry = _axis(time=time)

    def fail(*args, **kwargs):
        pytest.fail("q>=1 must refuse before the root proof")

    module = import_module("omnibias.core.verified.gaussian_affine_stationarity")
    monkeypatch.setattr(module, "certify_gaussian_affine_inverse", fail)
    assert certify_gaussian_affine_stationarity(geometry, (0, 0), (0, 1)) is None


@pytest.mark.parametrize("limits,target", [
    (RidgeBudget(max_matrix_entries=3000), 0),
    (RidgeBudget(max_elimination_work=10000), 0),
    (RidgeBudget(max_coefficient_bits=64), 10**100),
])
def test_aggregate_resource_refusal_precedes_composed_proof(limits, target, monkeypatch):
    geometry = _compile(GaussianHeatMap(((0,),), (0,), budget=limits), (), ())

    def fail(*args, **kwargs):
        pytest.fail("aggregate preflight must precede the root proof")

    module = import_module("omnibias.core.verified.gaussian_affine_stationarity")
    monkeypatch.setattr(module, "certify_gaussian_affine_inverse", fail)
    assert certify_gaussian_affine_stationarity(geometry, (target,), (0,)) is None


def test_nonfinite_multiplier_enclosure_is_inconclusive():
    source = GaussianHeatMap(((0,),), (0,))
    geometry = _compile(source, ((F(1, 10**200),),), (0,))
    assert certify_gaussian_affine_stationarity(geometry, (10**200,), (0,)) is None
