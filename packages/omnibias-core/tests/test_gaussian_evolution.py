# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Uniform inverse evolution contracts, sharp edge cases, and proof provenance."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_evolution import (
    GaussianInverseEvolution,
    certify_gaussian_inverse_evolution,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval


def _model(**kwargs):
    return GaussianHeatMap(((-1,), (1,)), (0, F(1, 3)), time=2, **kwargs)


def test_scalar_point_interval_has_useful_width_and_global_spectral_bounds():
    result = certify_gaussian_inverse_evolution(_model(), (Interval.point(0.25),))
    assert result is not None
    assert result.spatial_enclosure[0].hi-result.spatial_enclosure[0].lo < 1e-10
    jacobian = result.target_jacobian[0][0]
    assert 2 <= jacobian.lo <= jacobian.hi <= F(8, 3)
    assert jacobian.hi-jacobian.lo < 1e-9
    assert result.diffusion[0].hi-result.diffusion[0].lo < 1e-9
    assert result.diffusion_norm_bound == F(1, 6)
    assert result.time_derivative[0].hi < 0.25  # Asymmetric posterior gives a nonzero correction.


def test_wider_target_box_covers_its_point_certificates_with_independent_hull_constraint():
    model = _model()
    wide = certify_gaussian_inverse_evolution(model, (Interval(-0.125, 0.5),))
    assert wide is not None
    assert wide.spatial_enclosure[0].lo >= -1.25 and wide.spatial_enclosure[0].hi <= 2
    for v in (-0.125, 0, 0.0625, 0.25, 0.5):
        point = certify_gaussian_inverse_evolution(model, (Interval.point(v),))
        assert point is not None
        for broad, narrow in ((wide.spatial_enclosure[0], point.spatial_enclosure[0]),
                              (wide.target_jacobian[0][0], point.target_jacobian[0][0]),
                              (wide.time_derivative[0], point.time_derivative[0])):
            assert broad.lo <= narrow.lo <= narrow.hi <= broad.hi


def test_zero_iteration_coarse_root_is_still_honest_and_refinement_tightens():
    model = _model()
    coarse = certify_gaussian_inverse_evolution(model, (Interval.point(0.25),), max_iterations=0)
    fine = certify_gaussian_inverse_evolution(model, (Interval.point(0.25),))
    assert coarse is not None and fine is not None
    assert coarse.root_certificate is not None and fine.root_certificate is not None
    assert coarse.root_certificate.error_bound > fine.root_certificate.error_bound
    assert coarse.diffusion[0].lo <= fine.diffusion[0].lo <= fine.diffusion[0].hi <= coarse.diffusion[0].hi


def test_centered_symmetric_family_has_zero_evolution_correction():
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), time=2)
    result = certify_gaussian_inverse_evolution(model, (Interval.point(0.0),))
    assert result is not None
    assert result.diffusion[0].lo <= 0 <= result.diffusion[0].hi
    assert result.time_derivative[0].lo <= 0 <= result.time_derivative[0].hi
    assert result.diffusion[0].hi-result.diffusion[0].lo < 1e-11
    assert F(result.target_jacobian[0][0].lo) <= F(8, 3) <= F(result.target_jacobian[0][0].hi)


def test_affine_family_is_exact_without_transcendental_backend(monkeypatch):
    model = GaussianHeatMap(((2, -1), (2, -1)), (0, 10**400), time=F(3, 2))
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    box = (Interval(-2, 1), Interval(0, 2))
    result = certify_gaussian_inverse_evolution(model, box)
    assert result is not None
    assert result.root_certificate is None
    assert result.spatial_enclosure == (Interval(-1, 3.5), Interval(-1, 2))
    assert result.target_jacobian == ((Interval.point(1.5), Interval.point(0)), (Interval.point(0), Interval.point(1.5)))
    assert result.time_derivative == box
    assert result.diffusion == result.third_trace == (Interval.point(0), Interval.point(0))
    assert result.diffusion_norm_bound == 0
    assert result.certificate["meta"]["transcend_backend"] == "not_used"


def test_nonaffine_case_requires_rigorous_backend(monkeypatch):
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    with pytest.raises(RuntimeError):
        certify_gaussian_inverse_evolution(_model(), (Interval.point(0),))


def test_fresh_proof_preserves_sticky_fallback_and_snapshots_root(monkeypatch):
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    result = certify_gaussian_inverse_evolution(_model(), (Interval(0, 0.125),))
    assert result is not None and result.root_certificate is not None
    assert transcend.libm_fallback_used()
    assert result.certificate["meta"]["transcend_backend"] == "mpmath"
    assert verify_certificate_digest(result.certificate)
    saved = deepcopy(result.certificate)
    result.root_certificate.certificate["payload"]["error_bound"] = 1e99
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    assert result.certificate["payload"]["time"] == [2, 1]
    assert result.certificate["payload"]["biases"] == [[0, 1], [1, 3]]


@pytest.mark.parametrize("time", (F(1, 2), F(1, 4)))
def test_unproved_contraction_regimes_are_refused(time):
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), time=time)
    assert certify_gaussian_inverse_evolution(model, (Interval.point(0),)) is None


@pytest.mark.parametrize("kwargs,error", [
    ({"max_iterations": True}, TypeError), ({"max_iterations": 1.5}, TypeError),
    ({"max_iterations": -1}, ValueError), ({"absolute_tolerance": 0}, ValueError),
    ({"absolute_tolerance": float("nan")}, ValueError),
])
def test_invalid_controls_raise(kwargs, error):
    with pytest.raises(error):
        certify_gaussian_inverse_evolution(_model(), (Interval.point(0),), **kwargs)


@pytest.mark.parametrize("box,error", [
    (None, TypeError), ((0.0,), TypeError), ((), ValueError),
    ((Interval(0, 1), Interval(0, 1)), ValueError),
    ((Interval(0, float("inf")),), ValueError),
])
def test_invalid_boxes_raise(box, error):
    with pytest.raises(error):
        certify_gaussian_inverse_evolution(_model(), box)


def test_factory_record_is_immutable():
    with pytest.raises(TypeError):
        GaussianInverseEvolution()
    with pytest.raises(TypeError):
        certify_gaussian_inverse_evolution(None, (Interval(0, 1),))
    result = certify_gaussian_inverse_evolution(_model(), (Interval.point(0),))
    assert result is not None
    with pytest.raises(FrozenInstanceError):
        result.diffusion_norm_bound = F(0)


@pytest.mark.parametrize("limits,target", [
    (RidgeBudget(max_matrix_entries=500), 0),
    (RidgeBudget(max_elimination_work=2500), 0),
    (RidgeBudget(max_coefficient_bits=64), 1e100),
])
def test_exhausted_resources_are_inconclusive(limits, target):
    model = GaussianHeatMap(((1,), (1,)), (0, 1), budget=limits)
    assert certify_gaussian_inverse_evolution(model, (Interval.point(target),)) is None
