# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite joint face stability, including motion outside the old target box."""

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
from omnibias.core.verified.gaussian_entropy_bias_face import certify_gaussian_entropy_bias_face
from omnibias.core.verified.gaussian_entropy_bias_region import certify_gaussian_entropy_bias_region
from omnibias.core.verified.gaussian_entropy_face import certify_gaussian_entropy_face
from omnibias.core.verified.gaussian_entropy_joint_face import (
    GaussianEntropyJointFace,
    _joint_face_intact,
    _joint_face_source_slots,
    certify_gaussian_entropy_joint_face,
)
from omnibias.core.verified.gaussian_entropy_joint_region import (
    certify_gaussian_entropy_joint_region,
)
from omnibias.core.verified.gaussian_entropy_residual import certify_gaussian_entropy_residual
from omnibias.core.verified.gaussian_entropy_unique import certify_gaussian_entropy_unique
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _sources(*, centers=((0, 0),), biases=None, viscosity=1, time=1, target=(0, 0), rows=(), upper=(),
             reference=None, probabilities=None, bias_oscillation=F(1, 64), target_radius=F(1, 64)):
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(centers, (0,)*len(centers) if biases is None else biases, viscosity=viscosity, time=time)
    projector = compile_polyhedral_projector(model.dimension, rows, upper)
    assert projector is not None
    domain = certify_gaussian_posterior_domain(model, projector, target)
    assert domain is not None
    unique = certify_gaussian_entropy_unique(domain, reference=reference)
    assert unique is not None
    residual = certify_gaussian_entropy_residual(unique, (F(1, len(centers)),)*len(centers) if probabilities is None else probabilities)
    assert residual is not None
    face = certify_gaussian_entropy_face(residual)
    assert face is not None
    region = certify_gaussian_entropy_joint_region(unique, bias_oscillation=bias_oscillation, target_radius=target_radius)
    assert region is not None
    return face, region


def _result(face, region):
    result = certify_gaussian_entropy_joint_face(face, region)
    assert result is not None
    return result


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _contains(mp, bounds, point):
    for (lower, upper), value in zip(bounds, point, strict=True):
        assert _mp(mp, lower) <= value <= _mp(mp, upper)


def test_single_gaussian_target_motion_is_not_clipped_to_the_old_singleton_box():
    face, region = _sources(time=2, target_radius=F(1, 4), bias_oscillation=1000)
    result = _result(face, region)
    assert face.residual.unique.domain.physical_bounds == ((0, 0), (0, 0))
    assert result.optimizer_bounds == ((F(-1, 2), F(1, 2)),)*2
    assert result.projection_input_bounds == result.optimizer_bounds
    # Targets (3/20,1/5) and their negatives are on the Euclidean radius1/4
    # sphere. Their exact original Gaussian minimizers are twice the targets.
    for root in ((F(3, 10), F(2, 5)), (F(-3, 10), F(-2, 5))):
        for (lower, upper), value in zip(result.optimizer_bounds, root, strict=True):
            assert lower <= value <= upper
    assert result.optimizer_bounds == region.expanded_coordinate_bounds


def test_translated_nonbinary_sample_preserves_target_motion_outside_old_box():
    center = (F(1, 3), F(-2, 5))
    face, region = _sources(centers=(center,), target=(F(2, 7), F(-1, 9)), time=F(3, 2),
                            target_radius=F(1, 5), bias_oscillation=0)
    result = _result(face, region)
    base = face.residual.point
    actual = tuple(value+delta for value, delta in zip(base, (F(9, 50), F(6, 25)), strict=True))
    assert actual != base
    for pair, base_pair, value in zip(result.optimizer_bounds, face.optimizer_bounds, actual, strict=True):
        assert pair[0] <= value <= pair[1]
        assert not base_pair[0] <= value <= base_pair[1]


def test_full_rank_equalities_lock_x_while_true_normal_projection_input_moves():
    rows = ((1, 0), (-1, 0), (0, 1), (0, -1))
    face, region = _sources(rows=rows, upper=(0, 0, 0, 0), time=2, target_radius=F(1, 4), bias_oscillation=5)
    result = _result(face, region)
    assert result.optimizer_bounds == ((0, 0), (0, 0))
    assert result.projection_input_bounds == ((F(-1, 2), F(1, 2)),)*2
    assert all(value.lo > 0 for value in result.projection_multipliers)
    assert all(value.lo == value.hi == 0 for value in result.slacks)
    assert face.tangent_projector == ((0, 0), (0, 0))


def test_redundant_active_wall_allows_tangent_motion_outside_old_physical_box():
    face, region = _sources(target=(1, 0), rows=((1, 0), (2, 0)), upper=(0, 0),
                            target_radius=F(1, 4), bias_oscillation=F(1, 8))
    result = _result(face, region)
    assert face.residual.unique.domain.physical_bounds == ((0, 0), (0, 0))
    assert result.optimizer_bounds == ((0, 0), (F(-1, 4), F(1, 4)))
    assert all(value.lo > 0 for value in result.projection_multipliers)
    assert all(value.lo == value.hi == 0 for value in result.slacks)


def test_oblique_redundant_equality_preserves_tangent_space_for_joint_region():
    face, region = _sources(rows=((1, 1), (-1, -1), (2, 2)), upper=(0, 0, 0),
                            target_radius=F(1, 4))
    result = _result(face, region)
    # Project (3/20,-1/5), an admissible target perturbation, onto x+y=0.
    expected = (F(7, 40), F(-7, 40))
    for pair, value in zip(result.optimizer_bounds, expected, strict=True):
        assert pair[0] <= value <= pair[1]
    assert all(value.lo == value.hi == 0 for value in result.slacks)
    assert all(value.lo > 0 for value in result.projection_multipliers)


def test_zero_rows_do_not_block_motion_in_the_expanded_containment_box():
    face, region = _sources(rows=((0, 0), (0, 0)), upper=(0, 1), target_radius=F(1, 4))
    result = _result(face, region)
    assert result.optimizer_bounds == ((F(-1, 4), F(1, 4)),)*2
    assert result.slacks[0].lo == result.slacks[0].hi == 0
    assert result.slacks[1].lo == result.slacks[1].hi == 1


def test_both_zero_radii_replay_the_original_face_exactly():
    face, region = _sources(target=(1, F(1, 3)), rows=((1, 0), (2, 0)), upper=(0, 0),
                            target_radius=0, bias_oscillation=0)
    result = _result(face, region)
    assert region.displacement_bound == 0
    assert result.projection_input_bounds == face.projection_input_bounds
    assert result.optimizer_bounds == face.optimizer_bounds
    assert result.projection_multipliers == face.projection_multipliers
    assert result.slacks == face.slacks


def test_joint_target_and_bias_changes_match_original_gaussian_kkt_roots():
    mp = pytest.importorskip("mpmath")
    omega, radius = F(1, 32), F(1, 64)
    face, region = _sources(centers=((0, 0), (1, 0), (0, 1)), viscosity=2, target=(1, 0),
                            rows=((1, 0), (2, 0), (0, 1)), upper=(0, 0, 2),
                            bias_oscillation=omega, target_radius=radius)
    result = _result(face, region)
    with mp.workdps(100):
        for sign in (-1, 1):
            beta = (F(0), sign*omega/2, -sign*omega/2)
            target = (F(1), sign*radius)

            def posterior(y, beta=beta):
                logits = (_mp(mp, beta[0])-y*y/8, _mp(mp, beta[1])-(1+y*y)/8, _mp(mp, beta[2])-(y-1)**2/8)
                weights = tuple(mp.exp(value-max(logits)) for value in logits)
                return tuple(value/sum(weights) for value in weights)

            root = mp.findroot(lambda y, target=target: y-_mp(mp, target[1])-posterior(y)[2], mp.mpf(1)/3)
            probabilities = posterior(root)
            q = (_mp(mp, target[0])+probabilities[1], _mp(mp, target[1])+probabilities[2])
            assert q[0] > 0 and abs(q[1]-root) < mp.mpf("1e-90")
            _contains(mp, result.optimizer_bounds, (mp.mpf(0), root))
            _contains(mp, result.projection_input_bounds, q)
            original_multiplier = _mp(mp, face.witness_multipliers[0])+q[0]-_mp(mp, face.residual.projection_source[0])
            assert _mp(mp, result.projection_multipliers[0].lo) <= original_multiplier <= _mp(mp, result.projection_multipliers[0].hi)


def test_nonconvex_ambient_family_has_a_stable_finite_joint_region():
    mp = pytest.importorskip("mpmath")
    omega, radius = F(1, 128), F(1, 512)
    face, region = _sources(centers=((0, 0), (4, 0), (0, 4)), biases=(0, -20, -20),
                            target=(1, 0), rows=((1, 0), (2, 0)), upper=(0, 0), probabilities=(1, 0, 0),
                            bias_oscillation=omega, target_radius=radius)
    result = _result(face, region)
    assert face.residual.unique.domain.model.contraction_bound == 4
    with mp.workdps(100):
        root = mp.findroot(lambda y: y-_mp(mp, radius)-4*mp.exp(-24+_mp(mp, omega)+2*y)
                           /(1+mp.exp(-24)+mp.exp(-24+_mp(mp, omega)+2*y)), _mp(mp, radius))
        _contains(mp, result.optimizer_bounds, (mp.mpf(0), root))
    assert all(value.lo > 0 for value in result.projection_multipliers)


@pytest.mark.parametrize("base_target", [F(1, 100), F(-1, 100)])
def test_target_driven_face_switch_refuses_while_global_uniqueness_survives(base_target):
    face, region = _sources(centers=((0,),), target=(base_target,), rows=((1,),), upper=(0,),
                            bias_oscillation=0, target_radius=F(1, 8))
    assert region.contraction_bound == 0
    assert certify_gaussian_entropy_joint_face(face, region) is None
    # The exact single-Gaussian optimizer min(v,0) changes active set in the
    # declared target interval, on either side of the base target.
    assert min(base_target-F(1, 8), 0) < 0 == min(base_target+F(1, 8), 0)


def test_both_public_face_factories_reuse_one_propagation_kernel(monkeypatch):
    bias_module = import_module("omnibias.core.verified.gaussian_entropy_bias_face")
    joint_module = import_module("omnibias.core.verified.gaussian_entropy_joint_face")
    assert joint_module._propagate_face is bias_module._propagate_face
    original = bias_module._propagate_face
    calls = []

    def tracked(*args):
        calls.append(args[2])
        return original(*args)

    monkeypatch.setattr(bias_module, "_propagate_face", tracked)
    monkeypatch.setattr(joint_module, "_propagate_face", tracked)
    face, region = _sources(target=(1, 0), rows=((1, 0),), upper=(0,), target_radius=F(1, 4))
    bias_region = certify_gaussian_entropy_bias_region(face.residual.unique, F(1, 64))
    assert bias_region is not None and certify_gaussian_entropy_bias_face(face, bias_region) is not None
    assert _result(face, region)
    assert calls == [face.residual.unique.domain.physical_bounds, region.expanded_coordinate_bounds]
    assert calls[0] != calls[1]


def test_independent_copies_of_same_unique_proof_are_compatible():
    face, region = _sources()
    copied = deepcopy(region)
    assert copied.unique is not face.residual.unique
    assert _result(face, copied).region is copied


def test_different_unique_sources_refuse_even_with_identical_shapes():
    face, _ = _sources(target=(0, 0))
    _, other = _sources(target=(1, 0))
    assert certify_gaussian_entropy_joint_face(face, other) is None


def test_frozen_result_snapshots_both_sources_and_binds_exact_arrays():
    face, region = _sources(target=(1, 0), rows=((1, 0), (2, 0)), upper=(0, 0))
    result = _result(face, region)
    assert _joint_face_intact(result) and _joint_face_source_slots(result) > 0
    assert certificate_is_unconditional(result.certificate) and verify_certificate_digest(result.certificate)
    assert schema_errors_v1(result.certificate) == []
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    saved = deepcopy(result.certificate)
    with pytest.raises(TypeError):
        GaussianEntropyJointFace()
    with pytest.raises(FrozenInstanceError):
        result.optimizer_bounds = ()
    face.certificate["payload"]["type"] = "broken"
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    assert not _joint_face_intact(result)


@pytest.mark.parametrize("which", ["face", "region", "shared_unique", "snapshot", "payload"])
def test_damaged_nested_sources_refuse(which):
    face, region = deepcopy(_sources())
    if which == "face":
        face.certificate["payload"]["type"] = "damaged"
    elif which == "region":
        region.certificate["payload"]["type"] = "damaged"
    elif which == "shared_unique":
        face.residual.unique.certificate["payload"]["type"] = "damaged"
    else:
        if which == "snapshot":
            face.certificate["payload"]["residual_certificate"]["payload"]["type"] = "damaged"
        else:
            face.certificate["payload"] = []
        sealed = seal_certificate(face.certificate)
        face.certificate.clear()
        face.certificate.update(sealed)
    assert certify_gaussian_entropy_joint_face(face, region) is None


@pytest.mark.parametrize("which", ["face", "region"])
def test_conditional_sources_refuse(which):
    face, region = deepcopy(_sources())
    source = face if which == "face" else region
    source.certificate["meta"]["transcend_backend"] = "libm"
    sealed = seal_certificate(source.certificate)
    source.certificate.clear()
    source.certificate.update(sealed)
    assert certify_gaussian_entropy_joint_face(face, region) is None


@pytest.mark.parametrize("which", ["face", "region"])
@pytest.mark.parametrize("kind", ["storage", "work", "coefficient"])
def test_both_source_budgets_apply(which, kind):
    face, region = _sources(target=(F(1, 257), 0))
    region = deepcopy(region)
    model = face.residual.unique.domain.model if which == "face" else region.unique.domain.model
    limits = {"storage": RidgeBudget(max_matrix_entries=1), "work": RidgeBudget(max_elimination_work=1),
              "coefficient": RidgeBudget(max_coefficient_bits=8)}[kind]
    object.__setattr__(model, "budget", limits)
    assert certify_gaussian_entropy_joint_face(face, region) is None


def test_no_projection_solve_factorization_posterior_or_transcendental_query(monkeypatch):
    face, region = _sources(target=(1, 0), rows=((1, 0), (2, 0)), upper=(0, 0))

    def forbidden(*args, **kwargs):
        pytest.fail("joint face propagation performed a new numerical query")

    for suffix in ("polyhedral_projection", "gaussian_entropy_face", "gaussian_projection_minimum"):
        module = import_module("omnibias.core.verified."+suffix)
        monkeypatch.setattr(module, "_project_exact", forbidden)
    linear = import_module("omnibias.core._rational_linear")
    monkeypatch.setattr(linear, "factor_positive_definite", forbidden)
    monkeypatch.setattr(linear.RationalLDLT, "solve", forbidden)
    moments = import_module("omnibias.core.verified._gaussian_moments")
    monkeypatch.setattr(moments, "_posterior_moments", forbidden)
    for name in ("exp_iv", "ln_iv", "tanh_iv"):
        monkeypatch.setattr(transcend, name, forbidden)
    assert _result(face, region)


@pytest.mark.parametrize("strict", [False, True])
def test_missing_backend_and_sticky_history_preserve_original_state(strict, monkeypatch):
    face, region = _sources()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_entropy_joint_face(face, region)
    assert transcend.strict_backend() is strict and not transcend.libm_fallback_used()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="history"):
        certify_gaussian_entropy_joint_face(face, region)
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used()


def test_invalid_record_types_raise_without_a_backend():
    with pytest.raises(TypeError):
        certify_gaussian_entropy_joint_face(None, None)
    face, _ = _sources()
    with pytest.raises(TypeError):
        certify_gaussian_entropy_joint_face(face, None)


def test_consumer_integrity_binds_all_output_arrays():
    result = _result(*_sources())
    for name in ("projection_input_bounds", "optimizer_bounds", "projection_multipliers", "slacks"):
        altered = deepcopy(result)
        altered.certificate["payload"][name] = {"damaged": True}
        sealed = seal_certificate(altered.certificate)
        altered.certificate.clear()
        altered.certificate.update(sealed)
        assert not _joint_face_intact(altered)
