# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent original KKT checks for projected logit-group equilibria."""

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
from omnibias.core.verified.gaussian_grouped_equilibrium import certify_gaussian_grouped_equilibrium
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_projected_equilibrium import (
    GaussianProjectedEquilibrium,
    certify_gaussian_projected_equilibrium,
)


def _face(source=None, inequalities=((1,),), bounds=(0,), active=(0,), *, equalities=(), rhs=()):
    if source is None:
        source = GaussianHeatMap(((0,),), (0,))
    base = restrict_gaussian_heat(source, equalities, rhs)
    assert base is not None
    result = compile_gaussian_polyhedral_face(base, inequalities, bounds, active)
    assert result is not None
    return result


def _asymmetric(*, budget=None, shift=0, common_bias=0, permutation=None):
    relative = ((1, 1), (3, 2), (1, -1), (3, -2), (3, 0))
    centers = tuple((shift+x, y) for x, y in relative)
    scores = (0, 2, 0, 2, 2)
    biases = tuple(common_bias+F(score)+F(x*x+y*y, 64) for score, (x, y) in zip(scores, relative, strict=True))
    if permutation is not None:
        centers = tuple(centers[i] for i in permutation)
        biases = tuple(biases[i] for i in permutation)
    source = GaussianHeatMap(centers, biases, time=16, budget=budget)
    face = _face(source, ((1, 0), (0, 1), (0, 2), (-1, 0)), (shift, 0, 0, 1-shift), (0,))
    return face, (0, 0), (shift, 0)


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _contains(interval, value):
    assert value >= interval.lo and value <= interval.hi


def test_different_normal_centroids_bind_original_model_and_keep_all_tight_rows():
    face, target, point = _asymmetric()
    result = certify_gaussian_projected_equilibrium(face, target, point)
    assert result is not None
    assert certify_gaussian_grouped_equilibrium(face, target, point) is None
    assert result.groups == ((0, 2), (1, 3, 4))
    assert result.group_logits == (0, 2)
    assert result.group_centroids == ((1, 0), (3, 0))
    assert result.group_centroid_offsets == ((0, 0), (2, 0))
    assert result.projected_centroid == (0, 0)
    assert result.centered_centers == ((0, 1), (0, 2), (0, -1), (0, -2), (0, 0))
    assert result.group_multipliers == ((F(1, 16),), (F(3, 16),))
    assert result.slacks == (0, 0, 0, 1)
    assert result.active_indices == (0, 1, 2)
    assert result.strong_indices == (0,) and result.weak_indices == (1, 2)
    assert result.equality_multipliers == ()
    assert result.inequality_multipliers[0].lo > 0
    assert all(value.lo == value.hi == 0 for value in result.inequality_multipliers[1:])
    assert not hasattr(result, "hessian")


def test_independent_original_exponentials_verify_weights_and_kkt():
    mp = pytest.importorskip("mpmath")
    face, target, point = _asymmetric()
    result = certify_gaussian_projected_equilibrium(face, target, point)
    assert result is not None
    source = face.base_restriction.source_model
    with mp.workdps(90):
        raw = [mp.exp(_mp(mp, bias)-sum((_mp(mp, x)-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))/64)
               for bias, center in zip(source.biases, source.centers, strict=True)]
        posterior = [value/sum(raw) for value in raw]
        for enclosure, exact in zip(result.posterior, posterior, strict=True):
            _contains(enclosure, exact)
        for group, enclosure in zip(result.groups, result.group_probabilities, strict=True):
            _contains(enclosure, sum(posterior[index] for index in group))
        _contains(result.group_probabilities[0], 2/(2+3*mp.exp(2)))
        mean = [sum(weight*_mp(mp, center[axis]) for weight, center in zip(posterior, source.centers, strict=True))
                for axis in range(2)]
        velocity = [(_mp(mp, point[axis])-mean[axis])/16 for axis in range(2)]
        _contains(result.inequality_multipliers[0], -velocity[0])
        assert abs(velocity[1]) < mp.mpf("1e-85")
        assert abs(mean[0]-1) > mp.mpf("0.1")
        assert abs(mean[0]-3) > mp.mpf("0.01")
        assert all(abs(_mp(mp, target[axis])-velocity[axis]-(mean[0]/16 if axis == 0 else 0)) < mp.mpf("1e-85")
                   for axis in range(2))


def test_first_appearance_order_and_original_mode_order_are_preserved():
    face, target, point = _asymmetric(permutation=(1, 0, 3, 2, 4))
    result = certify_gaussian_projected_equilibrium(face, target, point)
    assert result is not None
    assert result.groups == ((0, 2, 4), (1, 3)) and result.group_logits == (0, -2)
    assert result.group_centroids == ((3, 0), (1, 0))
    assert result.group_centroid_offsets == ((0, 0), (-2, 0))
    assert result.centered_centers == ((0, 2), (0, 1), (0, -2), (0, -1), (0, 0))


def test_groupwise_stationarity_is_required_before_weight_evaluation(monkeypatch):
    source = GaussianHeatMap(((1, 0), (1, 1)), (F(1, 32), 1+F(2, 32)), time=8)
    face = _face(source, ((1, 0),), (0,), (0,))
    module = import_module("omnibias.core.verified.gaussian_projected_equilibrium")

    def fail(*args):
        pytest.fail("nonstationary groups must refuse before posterior evaluation")

    monkeypatch.setattr(module, "_probabilities", fail)
    assert certify_gaussian_projected_equilibrium(face, (0, 0), (0, 0)) is None


def test_equality_multiplier_may_have_any_sign_and_selected_zero_is_exactly_weak():
    centers = ((-1, 0, 1), (-1, 0, -1), (2, 0, 2), (2, 0, -2))
    biases = tuple(F(index//2)+sum(F(value)**2 for value in center)/64 for index, center in enumerate(centers))
    source = GaussianHeatMap(centers, biases, time=16)
    face = _face(source, ((0, 1, 0), (0, 0, 1)), (0, 0), (0,), equalities=((1, 0, 0),), rhs=(0,))
    result = certify_gaussian_projected_equilibrium(face, (-1, 0, 0), (0, 0, 0))
    assert result is not None
    assert result.group_multipliers == ((F(-17, 16), 0), (F(-7, 8), 0))
    assert result.equality_multipliers[0].hi < 0
    assert result.strong_indices == () and result.weak_indices == (0, 1)
    assert all(value.lo == value.hi == 0 for value in result.inequality_multipliers)


@pytest.mark.parametrize("score,accepted", [(1, True), (-1, False), (F(1, 2**100), True)])
def test_mixed_sign_coefficients_can_have_an_exact_abel_positive_proof(score, accepted):
    centers = ((-1,), (1,))
    source = GaussianHeatMap(centers, (F(1, 16), score+F(1, 16)), time=4)
    result = certify_gaussian_projected_equilibrium(_face(source), (0,), (0,))
    assert (result is not None) is accepted
    if result is not None:
        assert result.group_multipliers == ((F(-1, 4),), (F(1, 4),))
        assert result.inequality_multipliers[0].lo >= 0
        assert result.strong_indices == (0,) and result.weak_indices == ()
        assert result.certificate["payload"]["selected_multiplier_sign_proofs"] == [
            {"inequality_index": 0, "kind": "abel_positive", "weighted_suffixes": [[0, 1], [1, 4]]}]


def test_arbitrarily_close_distinct_logits_are_strong_but_exact_collapse_is_weak():
    epsilon = F(1, 10**1000)
    source = GaussianHeatMap(((-1,), (1,)), (F(1, 4), F(1, 4)+epsilon))
    result = certify_gaussian_projected_equilibrium(_face(source), (0,), (0,))
    assert result is not None
    assert result.group_logits == (0, epsilon) and result.groups == ((0,), (1,))
    assert result.group_multipliers == ((-1,), (1,))
    assert result.inequality_multipliers[0].lo == 0
    assert result.strong_indices == (0,) and result.weak_indices == ()
    assert result.certificate["payload"]["selected_multiplier_sign_proofs"][0]["weighted_suffixes"] == [[0, 1], [1, 1]]
    collapsed = GaussianHeatMap(((-1,), (1,)), (F(1, 4), F(1, 4)))
    weak = certify_gaussian_projected_equilibrium(_face(collapsed), (0,), (0,))
    assert weak is not None
    assert weak.groups == ((0, 1),) and weak.group_multipliers == ((0,),)
    assert weak.strong_indices == () and weak.weak_indices == (0,)
    assert weak.certificate["payload"]["selected_multiplier_sign_proofs"][0]["kind"] == "exact_zero"
    negative = GaussianHeatMap(((-1,), (1,)), (F(1, 4), F(1, 4)-epsilon))
    assert certify_gaussian_projected_equilibrium(_face(negative), (0,), (0,)) is None


def test_abel_suffixes_include_group_cardinalities():
    epsilon = F(1, 2**100)
    # Ignoring cardinality would falsely apply suffixes (0,1) to (-1,+1).
    negative = GaussianHeatMap(((-1,), (-1,), (1,)), (F(1, 4), F(1, 4), F(1, 4)+epsilon))
    assert certify_gaussian_projected_equilibrium(_face(negative), (0,), (0,)) is None
    positive = GaussianHeatMap(((-2,), (1,), (1,)), (F(1, 2), F(1, 8)+epsilon, F(1, 8)+epsilon), time=2)
    result = certify_gaussian_projected_equilibrium(_face(positive), (0,), (0,))
    assert result is not None
    assert result.groups == ((0,), (1, 2))
    assert result.group_multipliers == ((-1,), (F(1, 2),))
    assert result.certificate["payload"]["selected_multiplier_sign_proofs"][0]["weighted_suffixes"] == [[0, 1], [1, 1]]


def test_abel_order_is_sorted_by_exact_logits_and_keeps_first_appearance_metadata():
    source = GaussianHeatMap(((1,), (-1,)), (F(1, 4)+F(1, 2**100), F(1, 4)))
    result = certify_gaussian_projected_equilibrium(_face(source), (0,), (0,))
    assert result is not None
    assert result.groups == ((0,), (1,)) and result.group_logits == (0, -F(1, 2**100))
    assert result.certificate["payload"]["ascending_group_logit_order"] == [1, 0]
    assert result.certificate["payload"]["selected_multiplier_sign_proofs"][0]["weighted_suffixes"] == [[0, 1], [1, 1]]


def test_strict_interval_fallback_can_certify_when_abel_suffix_test_is_inconclusive():
    source = GaussianHeatMap(((-2,), (1,)), (F(1, 2), 2+F(1, 8)), time=2)
    result = certify_gaussian_projected_equilibrium(_face(source), (0,), (0,))
    assert result is not None
    assert result.group_multipliers == ((-1,), (F(1, 2),))
    assert result.inequality_multipliers[0].lo > 0 and result.strong_indices == (0,)
    assert result.certificate["payload"]["selected_multiplier_sign_proofs"] == [
        {"inequality_index": 0, "kind": "interval_positive", "weighted_suffixes": [[-1, 2], [1, 2]]}]


def test_abel_identity_matches_independent_high_precision_exponentials():
    mp = pytest.importorskip("mpmath")
    source = GaussianHeatMap(((-1,), (-1,), (2,), (3,)), (F(1, 16), F(1, 16), 1+F(1, 4), 3+F(9, 16)), time=4)
    result = certify_gaussian_projected_equilibrium(_face(source), (0,), (0,))
    assert result is not None
    payload = result.certificate["payload"]
    order = payload["ascending_group_logit_order"]
    suffixes = [F(*pair) for pair in payload["selected_multiplier_sign_proofs"][0]["weighted_suffixes"]]
    assert suffixes == [F(3, 4), F(5, 4), F(3, 4)]
    with mp.workdps(90):
        logits = [_mp(mp, result.group_logits[index]) for index in order]
        direct = sum(len(result.groups[index])*_mp(mp, result.group_multipliers[index][0])*mp.exp(_mp(mp, result.group_logits[index]))
                     for index in range(len(result.groups)))
        abel = _mp(mp, suffixes[0])*mp.exp(logits[0])+sum(_mp(mp, suffixes[k])*(mp.exp(logits[k])-mp.exp(logits[k-1]))
                                                                      for k in range(1, len(logits)))
        assert abs(direct-abel) < mp.mpf("1e-85") and abel > 0


def test_cardinality_weighting_obeys_intermediate_coefficient_bit_budget():
    source = GaussianHeatMap(((0,),)*4, (0,)*4, budget=RidgeBudget(max_coefficient_bits=64))
    face = _face(source)
    assert certify_gaussian_projected_equilibrium(face, (2**60,), (0,)) is not None
    # The multiplier itself fits; its cardinality-weighted Abel coefficient
    # needs 65 bits and must be refused even though a sign is apparent.
    assert F(2**62).numerator.bit_length() <= 64
    assert certify_gaussian_projected_equilibrium(face, (2**62,), (0,)) is None


def test_exact_positive_coefficient_proof_survives_weight_underflow():
    source = GaussianHeatMap(((0,), (1,)), (0, -10000+F(1, 4)))
    result = certify_gaussian_projected_equilibrium(_face(source), (0,), (0,))
    assert result is not None
    assert result.group_multipliers == ((0,), (1,))
    assert result.group_probabilities[1].lo == 0
    assert result.inequality_multipliers[0].lo == 0
    assert result.strong_indices == (0,) and result.weak_indices == ()


def test_negative_underflowed_mixture_is_never_declared_weak_or_feasible():
    source = GaussianHeatMap(((0,), (-1,)), (0, -10000+F(1, 4)))
    assert certify_gaussian_projected_equilibrium(_face(source), (0,), (0,)) is None


def test_single_group_exact_shortcut_matches_old_binding_and_requires_no_backend(monkeypatch):
    source = GaussianHeatMap(((0,), (1,), (3,)), (0, F(1, 32), F(9, 32)), time=8)
    face = _face(source)
    old = certify_gaussian_grouped_equilibrium(face, (0,), (0,))
    assert old is not None
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    result = certify_gaussian_projected_equilibrium(face, (0,), (0,))
    assert result is not None
    assert result.groups == ((0, 1, 2),) and result.group_logits == (0,)
    assert result.group_centroids == (old.centroid,)
    assert result.group_centroid_offsets == ((0,),)
    assert result.group_multipliers == (old.inequality_multipliers,)
    for value in result.posterior:
        _contains(value, F(1, 3))
    assert result.group_probabilities[0].lo == result.group_probabilities[0].hi == 1
    assert result.certificate["meta"]["transcend_backend"] == "not_used"


def test_empty_constraints_require_every_group_force_to_vanish():
    source = GaussianHeatMap(((1,), (3,), (0,), (4,)), (F(1, 32), F(9, 32), 1, F(3, 2)), time=8)
    face = _face(source, (), (), ())
    result = certify_gaussian_projected_equilibrium(face, (F(-1, 4),), (0,))
    assert result is not None
    assert result.group_centroids == ((2,), (2,))
    assert result.projected_centroid == (2,)
    assert result.group_multipliers == ((), ())
    assert result.equality_multipliers == result.inequality_multipliers == ()
    assert certify_gaussian_projected_equilibrium(face, (0,), (0,)) is None


def test_common_spatial_and_bias_gauges_never_require_absolute_squares_or_float_centroids():
    shift, common = 2**2500, 2**3000
    face, target, point = _asymmetric(shift=shift, common_bias=common)
    result = certify_gaussian_projected_equilibrium(face, target, point)
    assert result is not None
    assert result.group_centroids == ((shift+1, 0), (shift+3, 0))
    assert result.group_centroid_offsets == ((0, 0), (2, 0))
    assert result.group_logits == (0, 2)
    assert result.group_multipliers == ((F(1, 16),), (F(3, 16),))
    assert (shift*shift).bit_length() > face.base_restriction.source_model.budget.max_coefficient_bits


def test_global_base_contraction_not_selected_face_contraction_is_required():
    source = GaussianHeatMap(((-2,), (2,)), (0, 1))
    face = _face(source)
    assert face.geometry.restriction.model.contraction_bound == 0
    assert certify_gaussian_projected_equilibrium(face, (3,), (0,)) is None


def test_base_equalities_can_remove_ambient_negative_curvature():
    centers = ((10, 1), (10, -1), (-10, 2), (-10, -2))
    biases = tuple(F(index//2)+sum(F(value)**2 for value in center)/64 for index, center in enumerate(centers))
    source = GaussianHeatMap(centers, biases, time=16)
    face = _face(source, ((0, 1),), (0,), (0,), equalities=((1, 0),), rhs=(0,))
    assert source.contraction_bound > 1 and face.base_restriction.model.contraction_bound < 1
    result = certify_gaussian_projected_equilibrium(face, (0, 1), (0, 0))
    assert result is not None
    assert result.group_centroids == ((10, 0), (-10, 0))
    assert result.group_multipliers == ((F(5, 8), 1), (F(-5, 8), 1))
    assert result.inequality_multipliers[0].lo == result.inequality_multipliers[0].hi == 1
    assert result.strong_indices == (0,)


def test_tiny_exact_tangent_mismatch_is_not_tolerance_merged():
    epsilon = F(1, 2**100)
    source = GaussianHeatMap(((0, 0), (1, epsilon)), (0, 1+(1+epsilon**2)/8), time=2)
    face = _face(source, ((1, 0),), (0,), (0,))
    assert certify_gaussian_projected_equilibrium(face, (0, 0), (0, 0)) is None


def test_infeasible_points_and_negative_selected_multipliers_are_refused():
    assert certify_gaussian_projected_equilibrium(_face(), (1,), (1,)) is None
    assert certify_gaussian_projected_equilibrium(_face(active=()), (1,), (1,)) is None
    assert certify_gaussian_projected_equilibrium(_face(), (-1,), (0,)) is None


def test_factory_record_snapshots_inputs_and_embedded_evidence():
    with pytest.raises(TypeError):
        GaussianProjectedEquilibrium()
    face = _face()
    target, point = [1], [0]
    result = certify_gaussian_projected_equilibrium(face, target, point)
    assert result is not None
    with pytest.raises(FrozenInstanceError):
        result.groups = ()
    target[0] = point[0] = 99
    assert result.target == (1,) and result.point == (0,)
    assert verify_certificate_digest(result.certificate)
    assert schema_errors_v1(result.certificate) == []
    assert certificate_is_unconditional(result.certificate)
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    saved = deepcopy(result.certificate)
    face.certificate["payload"]["damaged"] = object()
    face.geometry.certificate["payload"]["damaged"] = object()
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    result.certificate["payload"]["group_centroids"] = [[999]]
    assert not verify_certificate_digest(result.certificate)


@pytest.mark.parametrize("dependency", ["face", "base", "geometry", "restriction"])
def test_damaged_dependency_is_inconclusive_before_posterior(dependency, monkeypatch):
    face, target, point = _asymmetric()
    record = {"face": face, "base": face.base_restriction, "geometry": face.geometry,
              "restriction": face.geometry.restriction}[dependency]
    record.certificate["payload"]["damaged"] = object()
    module = import_module("omnibias.core.verified.gaussian_projected_equilibrium")

    def fail(*args):
        pytest.fail("damaged dependency reached posterior evaluation")

    monkeypatch.setattr(module, "_probabilities", fail)
    assert certify_gaussian_projected_equilibrium(face, target, point) is None


@pytest.mark.parametrize("sticky", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_rigorous_posterior_preserves_strict_and_sticky_flags(sticky, strict, monkeypatch):
    pytest.importorskip("mpmath")
    face, target, point = _asymmetric()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    result = certify_gaussian_projected_equilibrium(face, target, point)
    assert result is not None
    assert result.certificate["meta"]["transcend_backend"] == "mpmath"
    assert transcend.libm_fallback_used() is sticky and transcend.strict_backend() is strict


@pytest.mark.parametrize("strict", [False, True])
def test_missing_rigorous_backend_raises_and_keeps_flags(strict, monkeypatch):
    face, target, point = _asymmetric()
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_projected_equilibrium(face, target, point)
    assert transcend.libm_fallback_used() and transcend.strict_backend() is strict


def test_one_group_retains_shared_strict_sealing_policy(monkeypatch):
    face = _face()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_projected_equilibrium(face, (1,), (0,))


def test_posterior_arithmetic_failure_is_inconclusive_and_restores_flags(monkeypatch):
    face, target, point = _asymmetric()
    module = import_module("omnibias.core.verified.gaussian_projected_equilibrium")

    def fail(*args):
        assert transcend.strict_backend()
        raise ArithmeticError("test interval overflow")

    monkeypatch.setattr(module, "_probabilities", fail)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    assert certify_gaussian_projected_equilibrium(face, target, point) is None
    assert not transcend.strict_backend() and transcend.libm_fallback_used()


def test_factory_does_not_solve_factor_iterate_or_rebuild_the_model(monkeypatch):
    face, target, point = _asymmetric()

    def fail(*args, **kwargs):
        pytest.fail("projected equilibrium attempted a solve or model rebuild")

    for name, functions in (
        ("omnibias.core._rational_linear", ("factor_positive_definite", "solve_affine")),
        ("omnibias.core.verified.inverse_gaussian_heat", ("inverse_gaussian_heat", "certify_gaussian_inverse")),
        ("omnibias.core.verified.gaussian_affine", ("restrict_gaussian_heat",)),
    ):
        module = import_module(name)
        for function in functions:
            monkeypatch.setattr(module, function, fail)
    monkeypatch.setattr(GaussianHeatMap, "__init__", fail)
    assert certify_gaussian_projected_equilibrium(face, target, point) is not None


@pytest.mark.parametrize("index", [1, 2])
@pytest.mark.parametrize("bad,error", [(None, TypeError), ((), ValueError), ((0, 0), ValueError),
    ((True,), TypeError), ((0.0,), TypeError), ((complex(0),), TypeError)])
def test_exact_vector_contract_precedes_preflight(index, bad, error, monkeypatch):
    arguments = [_face(), (1,), (0,)]
    arguments[index] = bad

    def fail(*args):
        pytest.fail("invalid vector reached budget preflight")

    monkeypatch.setattr(RidgeBudget, "_entries", fail)
    with pytest.raises(error):
        certify_gaussian_projected_equilibrium(*arguments)


def test_invalid_face_raises():
    with pytest.raises(TypeError):
        certify_gaussian_projected_equilibrium(None, (0,), (0,))


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_preflight_precedes_dependency_processing_and_group_allocation(guard, monkeypatch):
    face, target, point = _asymmetric()
    module = import_module("omnibias.core.verified.gaussian_projected_equilibrium")

    def exhausted(*args):
        raise ValueError("test budget exhausted")

    def fail(*args):
        pytest.fail("preflight must precede dependencies and posterior")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(module, "_face_intact", fail)
    monkeypatch.setattr(module, "_probabilities", fail)
    assert certify_gaussian_projected_equilibrium(face, target, point) is None


def test_grouping_replay_mixing_and_evidence_share_one_work_meter(monkeypatch):
    face, target, point = _asymmetric()
    observed = []

    def bounded(self, count):
        observed.append(count)
        if count > observed[0]+25:
            raise ValueError("cumulative budget exhausted")

    monkeypatch.setattr(RidgeBudget, "_work", bounded)
    assert certify_gaussian_projected_equilibrium(face, target, point) is None
    assert observed[-1] > observed[0]+25
    assert all(a < b for a, b in zip(observed, observed[1:], strict=False))


def test_exact_coefficient_growth_and_unrepresentable_logits_refuse():
    face, target, point = _asymmetric(budget=RidgeBudget(max_coefficient_bits=64))
    assert certify_gaussian_projected_equilibrium(face, (10**100, 0), point) is None
    source = GaussianHeatMap(((0,), (1,)), (0, -(10**400)))
    assert certify_gaussian_projected_equilibrium(_face(source), (0,), (0,)) is None
