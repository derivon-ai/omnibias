# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stable projection faces checked with original KKT data and exact geometry."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    seal_certificate,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_entropy_face import (
    GaussianEntropyFace,
    _face_intact,
    _face_source_slots,
    certify_gaussian_entropy_face,
)
from omnibias.core.verified.gaussian_entropy_residual import certify_gaussian_entropy_residual
from omnibias.core.verified.gaussian_entropy_unique import certify_gaussian_entropy_unique
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _residual(*, centers=((0, 0),), biases=None, viscosity=1, time=1, target=(0, 0),
              rows=(), upper=(), reference=None, probabilities=None):
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(centers, (0,)*len(centers) if biases is None else biases,
                            viscosity=viscosity, time=time)
    projector = compile_polyhedral_projector(model.dimension, rows, upper)
    assert projector is not None
    domain = certify_gaussian_posterior_domain(model, projector, target)
    assert domain is not None
    unique = certify_gaussian_entropy_unique(domain, reference=reference)
    assert unique is not None
    weights = (F(1, len(centers)),)*len(centers) if probabilities is None else probabilities
    result = certify_gaussian_entropy_residual(unique, weights)
    assert result is not None
    return result


def _face(residual=None, **kwargs):
    result = certify_gaussian_entropy_face(_residual() if residual is None else residual, **kwargs)
    assert result is not None
    return result


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _contains(mp, bound, value):
    assert _mp(mp, bound.lo) <= value <= _mp(mp, bound.hi)


def _matmul(left, right):
    columns = tuple(zip(*right, strict=True))
    return tuple(tuple(sum(x*y for x, y in zip(row, column, strict=True)) for column in columns) for row in left)


def test_empty_active_face_has_identity_derivative_and_no_witness_trials(monkeypatch):
    residual = _residual(target=(F(1, 3), F(-2, 5)))
    module = import_module("omnibias.core.verified.gaussian_entropy_face")

    def forbidden(*args, **kwargs):
        pytest.fail("empty face performed a witness projection")

    monkeypatch.setattr(module, "_project_exact", forbidden)
    face = _face(residual)
    assert face.active_indices == face.basis_indices == ()
    assert face.normal_rows == face.gram_inverse == ()
    assert face.tangent_projector == ((1, 0), (0, 1))
    assert face.projection_multipliers == face.slacks == face.witness_multipliers == ()
    assert face.witness_epsilon == 0
    assert face.optimizer_bounds == ((F(1, 3), F(1, 3)), (F(-2, 5), F(-2, 5)))


def test_strictly_interior_point_retains_inactive_positive_slacks():
    residual = _residual(target=(F(1, 3), F(-2, 5)), rows=((1, 0), (-1, 0), (0, 1), (0, -1)), upper=(1, 1, 1, 1))
    face = _face(residual)
    assert face.active_indices == () and face.tangent_projector == ((1, 0), (0, 1))
    assert all(value.lo > 0 for value in face.slacks)
    assert all(value.lo == value.hi == 0 for value in face.projection_multipliers)


def test_redundant_active_halfspace_has_strict_full_row_witness():
    residual = _residual(target=(1, F(1, 3)), rows=((1, 0), (2, 0), (-1, 0)), upper=(0, 0, 2))
    face = _face(residual)
    assert face.active_indices == (0, 1) and face.basis_indices == (0,)
    assert face.normal_rows == ((1, 0),) and face.gram_inverse == ((1,),)
    assert face.tangent_projector == ((0, 0), (0, 1))
    assert face.witness_epsilon == F(1, 4)
    assert face.witness_multipliers == (F(1, 2), F(1, 4), 0)
    assert all(face.projection_multipliers[index].lo > 0 for index in face.active_indices)
    assert face.slacks[2].lo == face.slacks[2].hi == 2
    for index in face.active_indices:
        assert face.slacks[index].lo == face.slacks[index].hi == 0


def test_opposite_and_redundant_rows_certify_an_oblique_equality():
    rows = ((1, 1), (-1, -1), (2, 2))
    residual = _residual(target=(2, 1), rows=rows, upper=(0, 0, 0))
    face = _face(residual)
    assert face.active_indices == (0, 1, 2) and face.basis_indices == (0,)
    assert face.gram_inverse == ((F(1, 2),),)
    assert face.tangent_projector == ((F(1, 2), F(-1, 2)), (F(-1, 2), F(1, 2)))
    assert all(value.lo > 0 for value in face.projection_multipliers)
    assert residual.point == (F(1, 2), F(-1, 2))
    assert _matmul(tuple(tuple(F(x) for x in row) for row in rows), face.tangent_projector) == ((0, 0),)*3
    assert _matmul(face.tangent_projector, face.tangent_projector) == face.tangent_projector


@pytest.mark.parametrize("rows", [((1, 0), (0, 1), (2, 2)), ((1, 0), (-1, 0), (0, 1), (0, -1))])
def test_full_rank_normal_space_has_zero_projector_even_with_redundant_constraints(rows):
    face = _face(_residual(target=(1, 2), rows=rows, upper=(0,)*len(rows)))
    assert face.active_indices == tuple(range(len(rows))) and len(face.basis_indices) == 2
    assert face.tangent_projector == ((0, 0), (0, 0))
    assert all(value.lo > 0 for value in face.projection_multipliers)
    assert all(value.lo == value.hi == 0 for value in face.slacks)


def test_zero_constraint_rows_are_tight_but_do_not_remove_tangent_dimensions():
    face = _face(_residual(target=(1, 2), rows=((0, 0), (0, 0), (0, 0)), upper=(0, 0, 1)))
    assert face.active_indices == (0, 1) and face.basis_indices == ()
    assert face.witness_multipliers == (1, 1, 0)
    assert face.tangent_projector == ((1, 0), (0, 1))
    assert face.slacks[-1].lo == face.slacks[-1].hi == 1


@pytest.mark.parametrize("rows", [((1, 0),), ((1, 0), (2, 0)), ((1, 0), (0, 1))])
def test_weak_one_sided_projection_switch_refuses(rows):
    residual = _residual(rows=rows, upper=(0,)*len(rows))
    assert certify_gaussian_entropy_face(residual) is None


def test_witness_trial_limit_is_inconclusive_and_can_be_increased():
    residual = _residual(target=(F(1, 8), 0), rows=((1, 0),), upper=(0,))
    assert certify_gaussian_entropy_face(residual, max_trials=3) is None
    face = _face(residual, max_trials=4)
    assert face.witness_epsilon == F(1, 8)
    assert face.projection_multipliers[0].lo == face.projection_multipliers[0].hi == F(1, 8)


def test_inactive_slack_failure_refuses_a_wrong_sample_face():
    # This proposal is interior, while the actual symmetric unconstrained
    # root lies beyond x<=-1/4. A broad residual ball cannot certify interior.
    residual = _residual(centers=((-1,), (1,)), target=(0,), rows=((1,),), upper=(F(-1, 4),),
                         reference=(0,), probabilities=(1, 0))
    assert residual.point == (-1,)
    assert certify_gaussian_entropy_face(residual) is None


def test_true_normal_margin_can_fail_despite_a_strict_sample_witness():
    # At x=0 the true symmetric optimum is a weak wall switch. The vertex
    # weights give a strict q0>0 normal, but its posterior correction cancels.
    residual = _residual(centers=((-1,), (1,)), target=(0,), rows=((1,),), upper=(0,),
                         reference=(0,), probabilities=(0, 1))
    assert residual.point == (0,) and residual.projection_source == (1,)
    assert certify_gaussian_entropy_face(residual) is None


def test_original_nonconvex_gaussian_posterior_covariance_and_projection_kkt():
    mp = pytest.importorskip("mpmath")
    residual = _residual(centers=((0, 0), (4, 0), (0, 4)), biases=(0, -20, -20),
                         target=(1, 0), rows=((1, 0), (2, 0)), upper=(0, 0), probabilities=(1, 0, 0))
    face = _face(residual)
    assert residual.unique.domain.model.contraction_bound == 4
    assert face.tangent_projector == ((0, 0), (0, 1))
    with mp.workdps(100):
        root = mp.findroot(lambda y: y-4*mp.exp(-24+2*y)/(1+mp.exp(-24)+mp.exp(-24+2*y)), mp.mpf("1e-10"))
        centers = ((0, 0), (4, 0), (0, 4))
        logits = tuple(_mp(mp, b)-((mp.mpf(a[0]))**2+(root-a[1])**2)/4
                       for b, a in zip((0, -20, -20), centers, strict=True))
        weights = tuple(mp.exp(logit) for logit in logits)
        probabilities = tuple(weight/sum(weights) for weight in weights)
        mean = tuple(sum(p*a[j] for p, a in zip(probabilities, centers, strict=True)) for j in range(2))
        covariance = tuple(tuple(sum(p*(a[i]-mean[i])*(a[j]-mean[j])
                                     for p, a in zip(probabilities, centers, strict=True)) for j in range(2)) for i in range(2))
        for bound, probability in zip(face.posterior, probabilities, strict=True):
            _contains(mp, bound, probability)
        for row, values in zip(face.covariance, covariance, strict=True):
            for bound, value in zip(row, values, strict=True):
                _contains(mp, bound, value)
        for pair, value in zip(face.optimizer_bounds, (mp.mpf(0), root), strict=True):
            assert _mp(mp, pair[0]) <= value <= _mp(mp, pair[1])
        # Original q*=v+mean, projected onto x1<=0, has strict normal q1*.
        assert mean[0]+1 > 0 and abs(mean[1]-root) < mp.mpf("1e-90")
        exact_multipliers = (_mp(mp, face.witness_multipliers[0])+mean[0], _mp(mp, face.witness_multipliers[1]))
        assert abs(exact_multipliers[0]+2*exact_multipliers[1]-(1+mean[0])) < mp.mpf("1e-90")
        for bound, value in zip(face.projection_multipliers, exact_multipliers, strict=True):
            _contains(mp, bound, value)


def test_huge_translation_keeps_tight_exact_optimizer_box_before_float_centering():
    original = _residual(centers=((-1,), (1,)), target=(0,), reference=(0,), probabilities=(F(1, 3), F(2, 3)))
    shift = 10**100
    translated = _residual(centers=((shift-1,), (shift+1,)), biases=(10**1000,)*2,
                            target=(0,), reference=(shift,), probabilities=(F(1, 3), F(2, 3)))
    first, second = _face(original), _face(translated)
    assert second.optimizer_bounds == tuple((lower+shift, upper+shift) for lower, upper in first.optimizer_bounds)
    assert translated.enclosure[0].hi-translated.enclosure[0].lo > 10**80
    assert second.optimizer_bounds[0][1]-second.optimizer_bounds[0][0] < 2
    assert first.posterior == second.posterior and first.covariance == second.covariance


def test_nonbinary_equalities_keep_all_structural_slacks_exact_zero():
    rows = ((3, 0), (-6, 0), (0, 7), (0, -14))
    upper = (1, -2, -2, 4)
    residual = _residual(rows=rows, upper=upper)
    face = _face(residual)
    assert residual.point == (F(1, 3), F(-2, 7))
    assert face.tangent_projector == ((0, 0), (0, 0))
    assert all(value.lo == value.hi == 0 for value in face.slacks)
    assert all(value.lo > 0 for value in face.projection_multipliers)


def test_cached_gram_factors_are_reused_without_new_elimination(monkeypatch):
    residual = _residual(target=(2, 1), rows=((1, 1), (-1, -1), (2, 2)), upper=(0, 0, 0))
    linear = import_module("omnibias.core._rational_linear")
    projection = import_module("omnibias.core.verified.polyhedral_projection")

    def forbidden(*args, **kwargs):
        pytest.fail("face certification recomputed rank or a factorization")

    for module in (linear, projection):
        monkeypatch.setattr(module, "factor_positive_definite", forbidden)
        monkeypatch.setattr(module, "solve_affine", forbidden)
    monkeypatch.setattr(projection, "compile_polyhedral_projector", forbidden)
    assert _face(residual).basis_indices == (0,)


def test_record_integrity_immutability_and_source_snapshot():
    residual = _residual(target=(1, 0), rows=((1, 0),), upper=(0,))
    face = _face(residual)
    assert _face_intact(face) and _face_source_slots(face) > 0
    assert verify_certificate_digest(face.certificate) and certificate_is_unconditional(face.certificate)
    assert schema_errors_v1(face.certificate) == []
    assert "theorem_prover_verified" not in face.certificate["honesty"]
    saved = deepcopy(face.certificate)
    with pytest.raises(TypeError):
        GaussianEntropyFace()
    with pytest.raises(FrozenInstanceError):
        face.witness_epsilon = F(0)
    residual.certificate["payload"]["type"] = "damaged"
    assert face.certificate == saved and verify_certificate_digest(face.certificate)
    assert not _face_intact(face)
    assert certify_gaussian_entropy_face(residual) is None


@pytest.mark.parametrize("damage", ["residual", "unique", "domain", "projector", "snapshot", "payload"])
def test_damaged_residual_dependencies_refuse(damage):
    residual = deepcopy(_residual())
    if damage == "unique":
        residual.unique.certificate["payload"]["type"] = "damaged"
    elif damage == "domain":
        residual.unique.domain.certificate["payload"]["type"] = "damaged"
    elif damage == "projector":
        residual.unique.domain.projector.certificate["payload"]["type"] = "damaged"
    elif damage in ("snapshot", "payload"):
        if damage == "snapshot":
            residual.certificate["payload"]["unique_certificate"]["payload"]["type"] = "damaged"
        else:
            residual.certificate["payload"] = []
        changed = seal_certificate(residual.certificate)
        residual.certificate.clear()
        residual.certificate.update(changed)
    else:
        residual.certificate["payload"]["type"] = "damaged"
    assert certify_gaussian_entropy_face(residual) is None


def test_conditional_residual_cannot_be_promoted_to_a_stable_face():
    residual = deepcopy(_residual())
    residual.certificate["meta"]["transcend_backend"] = "libm"
    changed = seal_certificate(residual.certificate)
    residual.certificate.clear()
    residual.certificate.update(changed)
    assert certify_gaussian_entropy_face(residual) is None


@pytest.mark.parametrize("trials,error", [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError), (F(2), TypeError)])
def test_invalid_trial_metadata_raises(trials, error):
    with pytest.raises(error):
        certify_gaussian_entropy_face(_residual(), max_trials=trials)


def test_invalid_source_type_raises_without_a_backend():
    with pytest.raises(TypeError):
        certify_gaussian_entropy_face(None)


@pytest.mark.parametrize("strict", [False, True])
def test_missing_backend_and_sticky_history_preserve_state(strict, monkeypatch):
    residual = _residual()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_entropy_face(residual)
    assert transcend.strict_backend() is strict and not transcend.libm_fallback_used()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="history"):
        certify_gaussian_entropy_face(residual)
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used()


@pytest.mark.parametrize("kind", ["storage", "work", "trials", "coefficient"])
def test_aggregate_budget_refusal_before_numerical_queries(kind, monkeypatch):
    residual = _residual(target=(F(1, 257), 0))
    max_trials = 32
    if kind == "storage":
        limits = RidgeBudget(max_matrix_entries=1)
    elif kind == "work":
        limits = RidgeBudget(max_elimination_work=1)
    elif kind == "coefficient":
        limits = RidgeBudget(max_coefficient_bits=8)
    else:
        limits = RidgeBudget()
        max_trials = 10**100
    object.__setattr__(residual.unique.domain.model, "budget", limits)
    module = import_module("omnibias.core.verified.gaussian_entropy_face")

    def forbidden(*args, **kwargs):
        pytest.fail("resource refusal occurred after a projection/posterior query")

    monkeypatch.setattr(module, "_project_exact", forbidden)
    monkeypatch.setattr(module, "_probabilities", forbidden)
    assert certify_gaussian_entropy_face(residual, max_trials=max_trials) is None


def test_face_consumer_integrity_checks_all_new_geometry_and_moment_bindings():
    face = _face()
    for key in ("tangent_projector", "posterior", "optimizer_bounds", "witness_multipliers", "strong_convexity_bound"):
        damaged = deepcopy(face)
        damaged.certificate["payload"][key] = {"damaged": True}
        sealed = seal_certificate(damaged.certificate)
        damaged.certificate.clear()
        damaged.certificate.update(sealed)
        assert not _face_intact(damaged)
    damaged = deepcopy(face)
    damaged.certificate["payload"] = []
    sealed = seal_certificate(damaged.certificate)
    damaged.certificate.clear()
    damaged.certificate.update(sealed)
    assert not _face_intact(damaged)
