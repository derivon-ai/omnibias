# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Global complex inverse tubes, exact witnesses and independent complex roots."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_tube import (
    GaussianInverseTubeCertificate,
    certify_gaussian_inverse_tube,
)


def _scalar(**kwargs):
    return GaussianHeatMap(((-1,), (1,)), (0, 0), **kwargs)


def _mp(mp, value):
    value = Fraction(value)
    return mp.mpf(value.numerator) / value.denominator


def test_supplied_radius_has_exact_replayable_bounds():
    model = _scalar()
    tube = certify_gaussian_inverse_tube(model, spatial_radius=Fraction(1, 2))
    assert tube is not None and tube.model is model
    assert tube.spatial_radius == Fraction(1, 2)
    assert tube.target_radius == Fraction(11, 45)
    assert tube.complex_contraction_bound == Fraction(8, 15)
    assert tube.correction_bound_squared == Fraction(64, 15)
    assert tube.reference_center == (Fraction(-1),)
    payload = tube.certificate["payload"]
    assert payload["default_radius_denominator_upper"] is None
    assert Fraction(*payload["phase_half_width_squared_bound"]) == Fraction(1, 16)
    assert Fraction(*payload["imaginary_mean_factor"]) == Fraction(23, 45)


@pytest.mark.parametrize("q", [Fraction(1, 100), Fraction(1, 2), Fraction(99, 100), Fraction(9999, 10000)])
def test_default_radius_is_checked_and_tube_improves_local_ball_target_radius(q):
    model = _scalar(time=1/(2*q))
    tube = certify_gaussian_inverse_tube(model)
    assert tube is not None
    payload = tube.certificate["payload"]
    upper = Fraction(*payload["default_radius_denominator_upper"])
    assert upper**2 >= model.diameter_squared*(1-q)
    assert tube.spatial_radius == 2*model.viscosity*model.time*(1-q)/upper
    beta = model.diameter_squared*tube.spatial_radius**2/(16*model.viscosity**2*model.time**2)
    assert beta <= (1-q)/4 < 1-q
    assert tube.complex_contraction_bound == q/(1-beta) < 1
    assert tube.target_radius > (1-tube.complex_contraction_bound)*tube.spatial_radius/model.time


def test_scalar_branch_points_and_other_roots_do_not_enter_certified_tube():
    mp = pytest.importorskip("mpmath")
    tube = certify_gaussian_inverse_tube(_scalar())
    with mp.workdps(90):
        r, bound = _mp(mp, tube.spatial_radius), _mp(mp, tube.target_radius)
        # The first imaginary critical point of U=x-tanh(x/2) is i*pi/2.
        critical_x = mp.j*mp.pi/2
        critical_v = critical_x-mp.tanh(critical_x/2)
        assert abs(1-mp.sech(critical_x/2)**2/2) < mp.mpf("1e-80")
        assert bound < abs(critical_v) and r < abs(critical_x)
        # U(x)=0 also has nonreal roots: uniqueness must remain tube-relative.
        other_y = mp.findroot(lambda y: y-mp.tan(y/2), (mp.mpf(2), mp.mpf("2.7")))
        assert abs(other_y-mp.tan(other_y/2)) < mp.mpf("1e-80")
        assert other_y > r
        for real in (-1000, -2, 0, 3, 1000):
            target = mp.mpf(real)+mp.j*bound*mp.mpf("0.99")
            root = mp.findroot(lambda x, target=target: x-mp.tanh(x/2)-target, target)
            assert abs(mp.im(root)) < r
            correction = abs(root-target+1)**2
            assert correction <= _mp(mp, tube.correction_bound_squared)


def test_coupled_original_family_complex_roots_obey_tube_and_correction():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0, 0), (1, 0), (0, 2)),
                            (Fraction(2, 3), Fraction(-3, 7), Fraction(5, 11)),
                            viscosity=Fraction(3, 2), time=Fraction(5, 4))
    tube = certify_gaussian_inverse_tube(model)
    assert tube is not None
    with mp.workdps(90):
        nu, time = _mp(mp, model.viscosity), _mp(mp, model.time)
        r, bound = _mp(mp, tube.spatial_radius), _mp(mp, tube.target_radius)
        centers = [tuple(_mp(mp, a) for a in row) for row in model.centers]
        biases = [_mp(mp, b) for b in model.biases]
        for real in ((0, 0), (20, -30), (-10, 40)):
            target = (mp.mpc(real[0], bound*mp.mpf("0.54")),
                      mp.mpc(real[1], -bound*mp.mpf("0.72")))

            def residual(x, y, target=target):
                # Direct complex Gaussian evaluation, independent of the proof.
                weights = [mp.exp(b-((x-a[0])**2+(y-a[1])**2)/(4*nu*time))
                           for a, b in zip(centers, biases, strict=True)]
                total = sum(weights)
                mean = tuple(sum(p*a[axis] for p, a in zip(weights, centers, strict=True))/total
                             for axis in range(2))
                return (x-mean[0])/time-target[0], (y-mean[1])/time-target[1]

            root = mp.findroot(residual, tuple(time*v for v in target))
            assert mp.sqrt(sum(mp.im(x)**2 for x in root)) < r
            correction = sum(abs(root[axis]-time*target[axis]-centers[0][axis])**2 for axis in range(2))
            assert correction <= _mp(mp, tube.correction_bound_squared)
            assert max(abs(value) for value in residual(*root)) < mp.mpf("1e-75")


def test_rational_tangent_bound_controls_imaginary_mean_up_to_spatial_boundary():
    mp = pytest.importorskip("mpmath")
    tube = certify_gaussian_inverse_tube(_scalar(), spatial_radius=1)
    assert tube is not None
    payload = tube.certificate["payload"]
    factor = Fraction(*payload["imaginary_mean_factor"])
    with mp.workdps(90):
        radius = _mp(mp, tube.spatial_radius)
        for fraction in (mp.mpf("0.01"), mp.mpf("0.5"), mp.mpf(1)):
            y = radius*fraction
            for real in (-20, -1, 0, 1, 20):
                imaginary_mean = abs(mp.im(mp.tanh((real+mp.j*y)/2)))
                assert imaginary_mean <= _mp(mp, factor)*radius
        assert mp.tan(radius/2) <= _mp(mp, factor)*radius


@pytest.mark.parametrize("centers,biases", [(((2, -3),), (10**100,)),
                                          (((2, -3), (2, -3)), (1, -100))])
def test_identical_centers_have_entire_affine_inverse(centers, biases):
    model = GaussianHeatMap(centers, biases, time=Fraction(3, 2))
    for requested in (None, Fraction(1, 100)):
        tube = certify_gaussian_inverse_tube(model, spatial_radius=requested)
        assert tube is not None
        assert tube.spatial_radius is None and tube.target_radius is None
        assert tube.complex_contraction_bound == tube.correction_bound_squared == 0
        assert tube.reference_center == (2, -3)
        assert tube.certificate["payload"]["entire_affine"] is True


@pytest.mark.parametrize("time", [Fraction(1, 2), Fraction(1, 4)])
def test_q_at_least_one_is_inconclusive(time):
    assert certify_gaussian_inverse_tube(_scalar(time=time)) is None


def test_supplied_spatial_radius_requires_strict_contraction_margin():
    model = _scalar(time=Fraction(2, 3))
    assert model.contraction_bound == Fraction(3, 4)
    # r=2/3 gives beta=1/4=1-q, where strict contraction is lost.
    assert certify_gaussian_inverse_tube(model, spatial_radius=Fraction(2, 3)) is None
    assert certify_gaussian_inverse_tube(model, spatial_radius=1) is None
    assert certify_gaussian_inverse_tube(model, spatial_radius=Fraction(3, 5)) is not None


@pytest.mark.parametrize("value,error", [(True, TypeError), (0.5, TypeError), (complex(1), TypeError),
                                         (0, ValueError), (-1, ValueError)])
def test_invalid_spatial_radius_raises(value, error):
    with pytest.raises(error):
        certify_gaussian_inverse_tube(_scalar(), spatial_radius=value)


def test_factory_only_type_checked_immutable_record():
    with pytest.raises(TypeError):
        GaussianInverseTubeCertificate()
    with pytest.raises(TypeError):
        certify_gaussian_inverse_tube(object())
    tube = certify_gaussian_inverse_tube(_scalar())
    with pytest.raises(FrozenInstanceError):
        tube.target_radius = Fraction(10)


def test_resource_and_intermediate_coefficient_refusals():
    assert certify_gaussian_inverse_tube(_scalar(budget=RidgeBudget(max_matrix_entries=100))) is None
    assert certify_gaussian_inverse_tube(_scalar(budget=RidgeBudget(max_elimination_work=1600))) is None
    model = _scalar(budget=RidgeBudget(max_coefficient_bits=64))
    assert certify_gaussian_inverse_tube(model) is None  # sqrt significand/scaling scratch bits
    assert certify_gaussian_inverse_tube(model, spatial_radius=Fraction(1, 2)) is not None
    # The supplied rational fits, but its squared intermediate does not.
    assert certify_gaussian_inverse_tube(model, spatial_radius=Fraction(1, 2**40)) is None
    assert certify_gaussian_inverse_tube(_scalar(), spatial_radius=2**5000) is None


def test_explicit_radius_can_avoid_default_binary64_sqrt_boundary():
    scale = 10**400
    model = GaussianHeatMap(((-scale,), (scale,)), (0, 0), viscosity=scale, time=scale)
    assert model.contraction_bound == Fraction(1, 2)
    # The default square-root upper bound is binary64 and cannot hold this
    # scale, but a supplied exact radius keeps the entire proof rational.
    assert certify_gaussian_inverse_tube(model) is None
    tube = certify_gaussian_inverse_tube(model, spatial_radius=scale)
    assert tube is not None and tube.target_radius == Fraction(4, 9)
    assert tube.correction_bound_squared == Fraction(16*scale**2, 3)


@pytest.mark.parametrize("sticky", [False, True])
def test_exact_proof_preserves_sticky_history_without_transcendental_backend(monkeypatch, sticky):
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    tube = certify_gaussian_inverse_tube(_scalar())
    assert tube is not None and tube.certificate["meta"]["transcend_backend"] == "not_used"
    assert certificate_is_unconditional(tube.certificate)
    assert schema_errors_v1(tube.certificate) == []
    assert transcend.libm_fallback_used() is sticky
    assert "theorem_prover_verified" not in tube.certificate["honesty"]


def test_exact_proof_preserves_shared_strict_sealing_refusal(monkeypatch):
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_inverse_tube(_scalar())
    assert transcend.libm_fallback_used() is True
    assert transcend.strict_backend() is True


@pytest.mark.parametrize("field", ["centers", "biases", "viscosity", "time", "reference_center",
                                   "diameter_squared", "real_contraction_bound", "spatial_radius", "target_radius",
                                   "complex_contraction_bound", "correction_bound_squared", "analytic_argument"])
def test_seal_binds_original_family_and_complete_proof(field):
    model = GaussianHeatMap(((-1,), (1,)), (Fraction(3, 7), Fraction(-2, 5)))
    tube = certify_gaussian_inverse_tube(model)
    assert tube is not None and verify_certificate_digest(tube.certificate)
    assert tube.certificate["payload"]["biases"] == [[3, 7], [-2, 5]]
    changed = deepcopy(tube.certificate)
    changed["payload"][field] = [123, 456]
    assert not verify_certificate_digest(changed)
