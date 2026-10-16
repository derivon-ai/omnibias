# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite bias face stability with independent perturbed original roots."""

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
from omnibias.core.verified.gaussian_entropy_bias_face import (
    GaussianEntropyBiasFace,
    _bias_face_intact,
    _bias_face_source_slots,
    certify_gaussian_entropy_bias_face,
)
from omnibias.core.verified.gaussian_entropy_bias_region import certify_gaussian_entropy_bias_region
from omnibias.core.verified.gaussian_entropy_face import certify_gaussian_entropy_face
from omnibias.core.verified.gaussian_entropy_residual import certify_gaussian_entropy_residual
from omnibias.core.verified.gaussian_entropy_unique import certify_gaussian_entropy_unique
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _sources(*, centers=((0, 0),), biases=None, viscosity=1, target=(0, 0), rows=(), upper=(),
             probabilities=None, reference=None, oscillation=F(1, 64)):
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(centers, (0,)*len(centers) if biases is None else biases, viscosity=viscosity)
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
    region = certify_gaussian_entropy_bias_region(unique, oscillation)
    assert region is not None
    return face, region


def _result(face, region):
    result = certify_gaussian_entropy_bias_face(face, region)
    assert result is not None
    return result


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _contains(mp, bound, value):
    assert _mp(mp, bound.lo) <= value <= _mp(mp, bound.hi)


def _inside(mp, bounds, point):
    for (lower, upper), value in zip(bounds, point, strict=True):
        assert _mp(mp, lower) <= value <= _mp(mp, upper)


@pytest.mark.parametrize("rows,upper,target", [
    ((), (), (F(1, 3), F(-2, 5))),
    (((1, 0), (2, 0)), (0, 0), (1, 2)),
    (((1, 1), (-1, -1), (2, 2)), (0, 0, 0), (2, 1)),
    (((1, 0), (-1, 0), (0, 1), (0, -1)), (0, 0, 0, 0), (1, 2)),
    (((0, 0), (0, 0)), (0, 1), (1, 2)),
])
def test_one_mode_arbitrary_bias_gauge_preserves_each_supported_face(rows, upper, target):
    face, region = _sources(rows=rows, upper=upper, target=target, oscillation=F(1000))
    result = _result(face, region)
    assert region.displacement_bound == 0
    assert result.projection_input_bounds == face.projection_input_bounds
    assert result.projection_multipliers == face.projection_multipliers
    assert result.slacks == face.slacks
    assert result.optimizer_bounds == tuple((x, x) for x in face.residual.point)


def test_zero_oscillation_replays_original_strict_margins():
    face, region = _sources(centers=((0, 0), (1, 0), (0, 1)), viscosity=2, target=(1, 0),
                            rows=((1, 0), (2, 0), (0, 1)), upper=(0, 0, 2), oscillation=F(0))
    result = _result(face, region)
    assert region.displacement_bound == 0
    assert result.projection_input_bounds == face.projection_input_bounds
    assert result.projection_multipliers == face.projection_multipliers
    assert result.slacks == face.slacks
    for new, old in zip(result.optimizer_bounds, face.optimizer_bounds, strict=True):
        assert old[0] <= new[0] <= new[1] <= old[1]


def test_redundant_active_face_contains_independent_perturbed_gaussian_roots():
    mp = pytest.importorskip("mpmath")
    omega = F(1, 32)
    face, region = _sources(centers=((0, 0), (1, 0), (0, 1)), viscosity=2, target=(1, 0),
                            rows=((1, 0), (2, 0), (0, 1)), upper=(0, 0, 2), oscillation=omega)
    result = _result(face, region)
    assert face.active_indices == (0, 1)
    assert result.slacks[0].lo == result.slacks[0].hi == result.slacks[1].lo == result.slacks[1].hi == 0
    assert result.slacks[2].lo > 0
    with mp.workdps(100):
        for shifts in ((0, 0, 0), (0, omega/2, -omega/2), (0, -omega/2, omega/2), (10**100,)*3):
            relative = tuple(_mp(mp, value-shifts[0]) for value in shifts)

            def posterior(y, relative=relative):
                logits = (relative[0]-y*y/8, relative[1]-(1+y*y)/8, relative[2]-(y-1)**2/8)
                weights = tuple(mp.exp(value-max(logits)) for value in logits)
                return tuple(value/sum(weights) for value in weights)

            root = mp.findroot(lambda y: y-posterior(y)[2], mp.mpf(1)/3)
            probabilities = posterior(root)
            _inside(mp, result.optimizer_bounds, (mp.mpf(0), root))
            q = (1+probabilities[1], probabilities[2])
            _inside(mp, result.projection_input_bounds, q)
            # Original normal KKT data: strict positive x1 multiplier,
            # exact x1=0, and y stationarity. The declared representation
            # moves only the independent row's base witness coefficient.
            assert q[0] > 1 and abs(root-q[1]) < mp.mpf("1e-90")
            multipliers = (_mp(mp, face.witness_multipliers[0])+q[0]-_mp(mp, face.residual.projection_source[0]),
                           _mp(mp, face.witness_multipliers[1]), mp.mpf(0))
            for interval, value in zip(result.projection_multipliers, multipliers, strict=True):
                _contains(mp, interval, value)
            _contains(mp, result.slacks[2], 2-root)


def test_nonconvex_original_family_keeps_face_through_finite_small_bias_changes():
    mp = pytest.importorskip("mpmath")
    omega = F(1, 128)
    face, region = _sources(centers=((0, 0), (4, 0), (0, 4)), biases=(0, -20, -20),
                            target=(1, 0), rows=((1, 0), (2, 0)), upper=(0, 0),
                            probabilities=(1, 0, 0), oscillation=omega)
    result = _result(face, region)
    assert face.residual.unique.domain.model.contraction_bound == 4
    with mp.workdps(100):
        for beta in (omega, -omega):
            root = mp.findroot(lambda y, beta=beta: y-4*mp.exp(-24+_mp(mp, beta)+2*y)
                               /(1+mp.exp(-24)+mp.exp(-24+_mp(mp, beta)+2*y)), mp.mpf("1e-10"))
            _inside(mp, result.optimizer_bounds, (mp.mpf(0), root))
    assert all(interval.lo > 0 for interval in result.projection_multipliers)


def test_oblique_equality_and_redundancy_keep_structural_slacks_zero():
    face, region = _sources(centers=((-1, 0), (1, 0)), reference=(0, 0),
                            rows=((1, 1), (-1, -1), (2, 2)), upper=(0, 0, 0), oscillation=F(1, 64))
    result = _result(face, region)
    assert all(interval.lo == interval.hi == 0 for interval in result.slacks)
    assert all(interval.lo > 0 for interval in result.projection_multipliers)
    assert face.tangent_projector == ((F(1, 2), F(-1, 2)), (F(-1, 2), F(1, 2)))


def test_lost_active_margin_refuses_and_original_optimum_releases_the_wall():
    mp = pytest.importorskip("mpmath")
    face, small = _sources(centers=((-1,), (1,)), target=(F(1, 100),), reference=(0,),
                           rows=((1,),), upper=(0,), oscillation=F(1, 4096))
    assert _result(face, small).projection_multipliers[0].lo > 0
    large = certify_gaussian_entropy_bias_region(face.residual.unique, F(1, 8))
    assert large is not None
    assert certify_gaussian_entropy_bias_face(face, large) is None
    with mp.workdps(100):
        root = mp.findroot(lambda x: x-mp.tanh(x/2-mp.mpf(1)/16)-mp.mpf(1)/100, -mp.mpf(1)/10)
        assert root < 0  # Changing the relative bias by -1/8 releases x<=0.


def test_lost_inactive_margin_refuses_and_original_optimum_hits_the_wall():
    face, region = _sources(centers=((-1,), (1,)), target=(0,), reference=(0,),
                            rows=((1,),), upper=(F(1, 100),), oscillation=F(1, 8))
    assert face.active_indices == () and region.displacement_bound > F(1, 100)
    assert certify_gaussian_entropy_bias_face(face, region) is None


def test_same_problem_can_be_bound_by_independent_deep_copies():
    face, region = _sources(target=(1, 0), rows=((1, 0),), upper=(0,))
    copied = deepcopy(region)
    assert copied.unique is not face.residual.unique
    assert _result(face, copied).region is copied


def test_different_base_certificates_refuse_even_when_shapes_match():
    face, _ = _sources(target=(1, 0), rows=((1, 0),), upper=(0,))
    _, other = _sources(target=(2, 0), rows=((1, 0),), upper=(0,))
    assert certify_gaussian_entropy_bias_face(face, other) is None


def test_new_result_is_frozen_and_snapshots_both_sources():
    face, region = _sources(target=(1, 0), rows=((1, 0), (2, 0)), upper=(0, 0))
    result = _result(face, region)
    assert _bias_face_intact(result) and _bias_face_source_slots(result) > 0
    assert verify_certificate_digest(result.certificate) and certificate_is_unconditional(result.certificate)
    assert schema_errors_v1(result.certificate) == []
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    saved = deepcopy(result.certificate)
    with pytest.raises(TypeError):
        GaussianEntropyBiasFace()
    with pytest.raises(FrozenInstanceError):
        result.optimizer_bounds = ()
    face.certificate["payload"]["type"] = "broken"
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    assert not _bias_face_intact(result)


@pytest.mark.parametrize("source", ["face", "region", "shared_unique", "snapshot", "payload"])
def test_damaged_bound_dependencies_refuse(source):
    face, region = deepcopy(_sources())
    if source == "face":
        face.certificate["payload"]["type"] = "broken"
    elif source == "shared_unique":
        face.residual.unique.certificate["payload"]["type"] = "broken"
    elif source == "region":
        region.certificate["payload"]["type"] = "broken"
    else:
        if source == "snapshot":
            face.certificate["payload"]["residual_certificate"]["payload"]["type"] = "broken"
        else:
            face.certificate["payload"] = []
        sealed = seal_certificate(face.certificate)
        face.certificate.clear()
        face.certificate.update(sealed)
    assert certify_gaussian_entropy_bias_face(face, region) is None


@pytest.mark.parametrize("which", ["face", "region"])
def test_conditional_source_refuses(which):
    face, region = deepcopy(_sources())
    source = face if which == "face" else region
    source.certificate["meta"]["transcend_backend"] = "libm"
    sealed = seal_certificate(source.certificate)
    source.certificate.clear()
    source.certificate.update(sealed)
    assert certify_gaussian_entropy_bias_face(face, region) is None


def test_factory_performs_no_projection_factorization_posterior_or_transcendental(monkeypatch):
    face, region = _sources(target=(1, 0), rows=((1, 0), (2, 0)), upper=(0, 0))

    def forbidden(*args, **kwargs):
        pytest.fail("finite face propagation performed a new numerical query")

    modules = (import_module("omnibias.core.verified.polyhedral_projection"),
               import_module("omnibias.core.verified.gaussian_entropy_face"),
               import_module("omnibias.core.verified.gaussian_projection_minimum"))
    for module in modules:
        if hasattr(module, "_project_exact"):
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
def test_strict_and_sticky_backend_contracts(strict, monkeypatch):
    face, region = _sources()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_entropy_bias_face(face, region)
    assert transcend.strict_backend() is strict and not transcend.libm_fallback_used()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="history"):
        certify_gaussian_entropy_bias_face(face, region)
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used()


@pytest.mark.parametrize("source", ["face", "region"])
@pytest.mark.parametrize("limit", ["storage", "work", "bits"])
def test_both_source_budget_sets_apply(source, limit):
    face, region = _sources(target=(F(1, 257), 0))
    region = deepcopy(region)
    model = face.residual.unique.domain.model if source == "face" else region.unique.domain.model
    budget = {"storage": RidgeBudget(max_matrix_entries=1), "work": RidgeBudget(max_elimination_work=1),
              "bits": RidgeBudget(max_coefficient_bits=8)}[limit]
    object.__setattr__(model, "budget", budget)
    assert certify_gaussian_entropy_bias_face(face, region) is None


def test_invalid_record_types_raise_without_a_backend():
    with pytest.raises(TypeError):
        certify_gaussian_entropy_bias_face(None, None)
    face, _ = _sources()
    with pytest.raises(TypeError):
        certify_gaussian_entropy_bias_face(face, None)


def test_consumer_integrity_binds_every_returned_array():
    result = _result(*_sources())
    for name in ("projection_input_bounds", "optimizer_bounds", "projection_multipliers", "slacks"):
        changed = deepcopy(result)
        changed.certificate["payload"][name] = {"damaged": True}
        sealed = seal_certificate(changed.certificate)
        changed.certificate.clear()
        changed.certificate.update(sealed)
        assert not _bias_face_intact(changed)
