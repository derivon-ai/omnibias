# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact common-centroid binding without fitted Gaussian probabilities."""

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
from omnibias.core.verified.gaussian_grouped_equilibrium import (
    GaussianGroupedEquilibrium,
    certify_gaussian_grouped_equilibrium,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_polyhedral_equilibrium import (
    certify_gaussian_polyhedral_equilibrium,
)


def _face(source=None, inequalities=((1,),), bounds=(0,), active=(0,), *, equalities=(), rhs=()):
    if source is None:
        source = GaussianHeatMap(((0,),), (0,))
    base = restrict_gaussian_heat(source, equalities, rhs)
    assert base is not None
    result = compile_gaussian_polyhedral_face(base, inequalities, bounds, active)
    assert result is not None
    return result


def _asymmetric(permutation=None):
    point = (F(3, 2), F(-1, 4))
    centers = ((2, 1), (1, 3), (0, 1), (1, -1), (1, 1))
    scores = (2, -1, 2, -1, -1)
    biases = tuple(F(score)+sum((x-a)**2 for x, a in zip(point, center, strict=True))/24
                   for score, center in zip(scores, centers, strict=True))
    if permutation is not None:
        centers = tuple(centers[i] for i in permutation)
        biases = tuple(biases[i] for i in permutation)
    source = GaussianHeatMap(centers, biases, viscosity=F(3, 2), time=4)
    face = _face(source, ((0, 1),), (point[1],), (0,), equalities=((1, 0),), rhs=(point[0],))
    return face, (F(-15, 8), F(43, 16)), point


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def test_unequal_group_cardinalities_and_weights_share_an_exact_centroid():
    face, target, point = _asymmetric()
    result = certify_gaussian_grouped_equilibrium(face, target, point)
    assert result is not None
    assert result.groups == ((0, 2), (1, 3, 4))
    assert result.group_logits == (0, -3)
    assert result.centroid == (1, 1)
    assert result.centered_centers == ((1, 0), (0, 2), (-1, 0), (0, -2), (0, 0))
    assert result.velocity == (F(1, 8), F(-5, 16))
    assert result.target != result.velocity
    assert result.equality_multipliers == (-2,) and result.inequality_multipliers == (3,)
    assert result.active_indices == result.strong_indices == (0,) and result.weak_indices == ()
    assert not hasattr(result, "posterior") and not hasattr(result, "hessian")


def test_independent_high_precision_original_weights_verify_only_the_point_mean():
    mp = pytest.importorskip("mpmath")
    face, target, point = _asymmetric()
    source = face.base_restriction.source_model
    result = certify_gaussian_grouped_equilibrium(face, target, point)
    assert result is not None
    with mp.workdps(90):
        weights = [mp.exp(_mp(mp, bias)-sum((_mp(mp, x)-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))/24)
                   for bias, center in zip(source.biases, source.centers, strict=True)]
        total = sum(weights)
        assert abs(weights[0]/weights[1]-mp.exp(3)) < mp.mpf("1e-85")
        assert abs(sum(weights[i] for i in result.groups[0])/total-mp.mpf("0.5")) > mp.mpf("0.1")
        for axis in range(2):
            mean = sum(weight*_mp(mp, center[axis]) for weight, center in zip(weights, source.centers, strict=True))/total
            assert abs(mean-_mp(mp, result.centroid[axis])) < mp.mpf("1e-85")
    # The equal-centroid grouping need not survive a change in the point.
    nearby = (point[0]+F(1, 100), point[1])
    unconstrained = _face(source, (), (), ())
    assert certify_gaussian_grouped_equilibrium(unconstrained, target, nearby) is None


def test_first_appearance_order_tracks_original_modes_not_sorted_logits():
    face, target, point = _asymmetric((1, 0, 3, 2, 4))
    result = certify_gaussian_grouped_equilibrium(face, target, point)
    assert result is not None
    assert result.groups == ((0, 2, 4), (1, 3))
    assert result.group_logits == (0, 3)
    assert result.centroid == (1, 1) and result.velocity == (F(1, 8), F(-5, 16))


def test_one_group_agrees_with_equal_logit_certificate():
    source = GaussianHeatMap(((0,), (1,), (3,)), (0, F(1, 32), F(9, 32)), time=8)
    face = _face(source, (), (), ())
    grouped = certify_gaussian_grouped_equilibrium(face, (F(-1, 6),), (0,))
    uniform = certify_gaussian_polyhedral_equilibrium(face, (F(-1, 6),), (0,))
    assert grouped is not None and uniform is not None
    assert grouped.groups == ((0, 1, 2),) and grouped.group_logits == (0,)
    assert grouped.centroid == (F(4, 3),)
    assert grouped.velocity == uniform.velocity
    assert grouped.equality_multipliers == uniform.equality_multipliers
    assert grouped.inequality_multipliers == uniform.inequality_multipliers
    assert grouped.strong_convexity_bound == uniform.strong_convexity_bound


def test_different_logits_at_identical_centers_need_no_weight_evaluation():
    source = GaussianHeatMap(((2,),)*3, (0, 10**100, F(-2, 7)), time=2)
    result = certify_gaussian_grouped_equilibrium(_face(source, (), (), ()), (F(1, 2),), (3,))
    assert result is not None
    assert result.groups == ((0,), (1,), (2,))
    assert result.group_logits == (0, 10**100, F(-2, 7))
    assert result.centroid == (2,) and result.centered_centers == ((0,),)*3
    assert result.velocity == (F(1, 2),)


@pytest.mark.parametrize("contrast", [F(1), F(1, 2**100)])
def test_unequal_group_centroids_are_refused_without_tolerance_merging(contrast):
    source = GaussianHeatMap(((-1,), (1,)), (0, contrast), time=2)
    assert certify_gaussian_grouped_equilibrium(_face(source), (1,), (0,)) is None


def test_common_spatial_and_bias_gauges_do_not_require_absolute_squares():
    shift, common = 2**2500, 2**3000
    source = GaussianHeatMap(((shift-1,), (shift-2,), (shift+1,), (shift+2,)),
                             (common+F(1, 64), common+F(17, 16), common+F(1, 64), common+F(17, 16)), time=16)
    result = certify_gaussian_grouped_equilibrium(_face(source, ((1,),), (shift,), (0,)), (1,), (shift,))
    assert result is not None
    assert result.groups == ((0, 2), (1, 3)) and result.group_logits == (0, 1)
    assert result.centroid == (shift,) and result.centered_centers == ((-1,), (-2,), (1,), (2,))
    assert result.velocity == (0,) and result.inequality_multipliers == (1,)
    assert (shift*shift).bit_length() > source.budget.max_coefficient_bits


def test_all_omitted_tight_rows_including_redundant_rows_are_weak():
    source = GaussianHeatMap(((0, 0),), (0,))
    face = _face(source, ((1, 0), (2, 0), (0, 1), (-1, 0)), (0, 0, 0, 1), (0,))
    result = certify_gaussian_grouped_equilibrium(face, (1, 0), (0, 0))
    assert result is not None
    assert result.active_indices == (0, 1, 2)
    assert result.strong_indices == (0,) and result.weak_indices == (1, 2)
    assert result.inequality_multipliers == (1, 0, 0, 0) and result.slacks == (0, 0, 0, 1)


def test_negative_multiplier_nonstationary_and_infeasible_points_are_refused():
    assert certify_gaussian_grouped_equilibrium(_face(), (-1,), (0,)) is None
    assert certify_gaussian_grouped_equilibrium(_face(), (1,), (1,)) is None
    assert certify_gaussian_grouped_equilibrium(_face(active=()), (1,), (1,)) is None
    face = _face(GaussianHeatMap(((0, 0),), (0,)), ((1, 0),), (0,), (0,))
    assert certify_gaussian_grouped_equilibrium(face, (1, 1), (0, 0)) is None


def test_global_base_contraction_is_required_even_at_a_singleton_face():
    source = GaussianHeatMap(((-2,), (2,)), (0, 0))
    face = _face(source)
    assert face.geometry.restriction.model.contraction_bound == 0
    assert certify_gaussian_grouped_equilibrium(face, (1,), (0,)) is None


def test_factory_only_record_snapshots_exact_inputs_and_dependencies():
    with pytest.raises(TypeError):
        GaussianGroupedEquilibrium()
    face = _face()
    target, point = [1], [0]
    result = certify_gaussian_grouped_equilibrium(face, target, point)
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
    face.base_restriction.certificate["payload"]["damaged"] = object()
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    result.certificate["payload"]["groups"] = [[99]]
    assert not verify_certificate_digest(result.certificate)


@pytest.mark.parametrize("dependency", ["face", "base", "geometry", "restriction"])
def test_damaged_dependency_is_inconclusive(dependency):
    face = _face()
    record = {"face": face, "base": face.base_restriction, "geometry": face.geometry,
              "restriction": face.geometry.restriction}[dependency]
    record.certificate["payload"]["damaged"] = object()
    assert certify_gaussian_grouped_equilibrium(face, (1,), (0,)) is None


@pytest.mark.parametrize("sticky", [False, True])
def test_exact_binding_uses_no_backend_or_solver_and_preserves_history(sticky, monkeypatch):
    face, target, point = _asymmetric()

    def fail(*args, **kwargs):
        pytest.fail("exact grouped binding attempted posterior evaluation or a solve")

    for module_name, names in (
        ("omnibias.core.verified.mean_map", ("_probabilities",)),
        ("omnibias.core.verified.inverse_gaussian_heat", ("inverse_gaussian_heat", "certify_gaussian_inverse")),
        ("omnibias.core._rational_linear", ("factor_positive_definite", "solve_affine")),
    ):
        module = import_module(module_name)
        for name in names:
            monkeypatch.setattr(module, name, fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    result = certify_gaussian_grouped_equilibrium(face, target, point)
    assert result is not None and result.certificate["meta"]["transcend_backend"] == "not_used"
    assert transcend.libm_fallback_used() is sticky and not transcend.strict_backend()


def test_shared_strict_sealing_policy_is_preserved(monkeypatch):
    face = _face()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_grouped_equilibrium(face, (1,), (0,))
    assert transcend.strict_backend()


@pytest.mark.parametrize("index", [1, 2])
@pytest.mark.parametrize("bad,error", [(None, TypeError), ((), ValueError), ((0, 0), ValueError),
    ((True,), TypeError), ((0.0,), TypeError), ((complex(0),), TypeError)])
def test_exact_vector_contract(index, bad, error):
    arguments = [_face(), (1,), (0,)]
    arguments[index] = bad
    with pytest.raises(error):
        certify_gaussian_grouped_equilibrium(*arguments)


def test_invalid_face_raises():
    with pytest.raises(TypeError):
        certify_gaussian_grouped_equilibrium(None, (0,), (0,))


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_preflight_precedes_dependency_processing_and_group_allocation(guard, monkeypatch):
    face = _face()

    def exhausted(*args):
        raise ValueError("test exact budget exhausted")

    def fail(*args):
        pytest.fail("preflight must precede grouped dependency processing")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    module = import_module("omnibias.core.verified.gaussian_grouped_equilibrium")
    monkeypatch.setattr(module, "_face_intact", fail)
    assert certify_gaussian_grouped_equilibrium(face, (1,), (0,)) is None


def test_input_and_logit_intermediate_bits_are_bounded():
    source = GaussianHeatMap(((-1,), (1,)), (0, 0), time=2, budget=RidgeBudget(max_coefficient_bits=64))
    face = _face(source, (), (), ())
    assert certify_gaussian_grouped_equilibrium(face, (10**100,), (0,)) is None
    assert certify_gaussian_grouped_equilibrium(face, (0,), (2**63,)) is None


def test_grouping_and_kkt_arithmetic_share_one_work_meter(monkeypatch):
    face, target, point = _asymmetric()
    observed = []

    def bounded(self, count):
        observed.append(count)
        if count > observed[0]+25:
            raise ValueError("cumulative budget exhausted")

    monkeypatch.setattr(RidgeBudget, "_work", bounded)
    assert certify_gaussian_grouped_equilibrium(face, target, point) is None
    assert observed[-1] > observed[0]+25
    assert all(a < b for a, b in zip(observed, observed[1:], strict=False))
