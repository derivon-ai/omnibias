# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Analytic inverse domains, exact coefficient rounding and genuine tail checks."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest
from omnibias.core.gaussian_jet import GaussianInverseSeries, gaussian_inverse_series
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_analytic import (
    GaussianSeriesCertificate,
    certify_gaussian_inverse_series,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap


def _scalar(order=7, **kwargs):
    return gaussian_inverse_series(
        ((-1,), (1,)), (Fraction(1, 2),) * 2, (0,),
        ((0,), (1,)) + ((0,),) * (order - 1), **kwargs,
    )


def _mp(mp, value):
    value = Fraction(value)
    return mp.mpf(value.numerator) / value.denominator


def test_explicit_radius_and_polynomial_growth_are_exact():
    series = _scalar()
    certificate = certify_gaussian_inverse_series(series)
    assert certificate is not None
    payload = certificate.certificate["payload"]
    denominator = Fraction(*payload["strip_denominator_upper"])
    assert denominator**2 >= 2
    assert certificate.spatial_radius == 1/denominator
    assert certificate.complex_contraction_bound == Fraction(4, 7)
    assert certificate.parameter_radius == 3*certificate.spatial_radius/7
    assert certificate.tail_bound(certificate.parameter_radius / 2) == certificate.spatial_radius/128
    assert Fraction(*payload["strip_denominator_squared"]) == 2
    assert Fraction(*payload["phase_half_width_squared_bound"]) == Fraction(1, 8)
    assert certificate.tail_bound(0) == 0
    assert certificate.series is series


def test_scalar_tail_contains_independent_implicit_roots_and_shrinks_with_order():
    mp = pytest.importorskip("mpmath")
    certificates = [certify_gaussian_inverse_series(_scalar(order)) for order in (3, 7, 11)]
    assert all(certificate is not None for certificate in certificates)
    with mp.workdps(90):
        for sign in (-1, 1):
            parameter = sign * certificates[0].parameter_radius / 2
            target = _mp(mp, parameter)
            root = mp.findroot(lambda x, target=target: x - mp.tanh(x / 2) - target, 2 * target)
            previous = None
            for certificate in certificates:
                enclosure = certificate.enclose(parameter)[0]
                assert mp.mpf(enclosure.lo) <= root <= mp.mpf(enclosure.hi)
                error = abs(root - _mp(mp, certificate.series.evaluate(parameter)[0]))
                tail = certificate.tail_bound(abs(parameter))
                assert error <= _mp(mp, tail)
                if previous is not None:
                    assert tail == previous / 16
                previous = tail


def test_coupled_polynomial_target_has_independent_heat_inverse_enclosures():
    mp = pytest.importorskip("mpmath")
    centers = ((0, 0), (1, 0), (0, 1))
    target = ((Fraction(-1, 3),) * 2, (Fraction(1, 2), Fraction(-1, 3)),
              (Fraction(1, 4), Fraction(1, 5))) + ((0, 0),) * 5
    series = gaussian_inverse_series(centers, (Fraction(1, 3),) * 3, (0, 0), target)
    certificate = certify_gaussian_inverse_series(series)
    assert certificate is not None
    with mp.workdps(90):
        for sign in (-1, 1):
            parameter = sign * certificate.parameter_radius / 2
            t = _mp(mp, parameter)
            wanted = tuple(sum(_mp(mp, row[axis]) * t**n for n, row in enumerate(target)) for axis in range(2))

            def residual(x, y, wanted=wanted):
                weights = (mp.mpf(1), mp.exp(x / 2), mp.exp(y / 2))
                total = sum(weights)
                return x - weights[1] / total - wanted[0], y - weights[2] / total - wanted[1]

            root = mp.findroot(residual, (t, -t))
            box = certificate.enclose(parameter)
            for axis in range(2):
                assert mp.mpf(box[axis].lo) <= root[axis] <= mp.mpf(box[axis].hi)


def test_supplied_radius_checks_all_polynomial_target_terms():
    series = gaussian_inverse_series(((-1,), (1,)), (Fraction(1, 2),) * 2, (0,),
                                     ((0,), (0,), (0,), (1,)))
    # The stronger strip proof gives r*(1-kappa) approximately0.303.
    # R=1/2 passes R^3=1/8, while R=1 fails despite vanishing v1 and v2.
    certificate = certify_gaussian_inverse_series(series, parameter_radius=Fraction(1, 2))
    assert certificate is not None and certificate.parameter_radius == Fraction(1, 2)
    assert certify_gaussian_inverse_series(series, parameter_radius=1) is None


def test_affine_family_is_entire_and_rational_evaluation_rounds_outward():
    series = gaussian_inverse_series(((Fraction(1, 3),),), (1,), (Fraction(1, 3),),
                                     ((0,), (Fraction(1, 7),)), time=Fraction(2, 3))
    certificate = certify_gaussian_inverse_series(series, parameter_radius=Fraction(1, 100))
    assert certificate is not None
    assert certificate.parameter_radius is None and certificate.spatial_radius is None
    assert certificate.tail_bound(10**40) == 0
    enclosure = certificate.enclose(Fraction(1, 10))[0]
    exact = Fraction(12, 35)
    assert Fraction(enclosure.lo) <= exact <= Fraction(enclosure.hi)
    assert enclosure.lo < enclosure.hi  # Non-binary rational center stays enclosed.


def test_constant_target_is_entire_even_for_distinct_centers():
    series = gaussian_inverse_series(((-1,), (1,)), (Fraction(1, 2),) * 2, (0,), ((0,),) * 5)
    certificate = certify_gaussian_inverse_series(series)
    assert certificate is not None and certificate.parameter_radius is None
    assert certificate.tail_bound(10**50) == 0
    assert certificate.enclose(-(10**50))[0].contains(0)


def test_equal_logits_factory_binds_an_existing_model_without_rounding_posterior():
    model = GaussianHeatMap(((0, 0), (1, 0), (0, 1)), (0, Fraction(1, 4), Fraction(1, 4)))
    series = GaussianInverseSeries.from_equal_logits(model, (0, 0), ((1, 0), (0, 0), (0, 0)))
    certificate = certify_gaussian_inverse_series(series)
    assert certificate is not None
    assert series.posterior == (Fraction(1, 3),) * 3
    assert series.target_coefficients[0] == (Fraction(-1, 3),) * 2


@pytest.mark.parametrize("sticky", [False, True])
def test_exact_proof_uses_no_transcendental_backend_and_preserves_provenance(monkeypatch, sticky):
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    certificate = certify_gaussian_inverse_series(_scalar())
    assert certificate is not None
    assert certificate.certificate["meta"]["transcend_backend"] == "not_used"
    assert certificate_is_unconditional(certificate.certificate)
    assert transcend.libm_fallback_used() is sticky
    assert schema_errors_v1(certificate.certificate) == []
    assert "theorem_prover_verified" not in certificate.certificate["honesty"]


@pytest.mark.parametrize("field", ["posterior", "base_point", "target_coefficients", "coefficients", "parameter_radius",
                                   "strip_denominator_squared", "strip_denominator_upper", "phase_half_width_squared_bound"])
def test_seal_binds_family_target_and_coefficients(field):
    certificate = certify_gaussian_inverse_series(_scalar())
    changed = deepcopy(certificate.certificate)
    changed["payload"][field] = [123, 456]
    assert not verify_certificate_digest(changed)


@pytest.mark.parametrize("value,error", [(True, TypeError), (0.1, TypeError), (0, ValueError), (-1, ValueError)])
def test_invalid_requested_radius_raises(value, error):
    with pytest.raises(error):
        certify_gaussian_inverse_series(_scalar(), parameter_radius=value)


def test_tail_and_evaluation_enforce_strict_analytic_domain():
    certificate = certify_gaussian_inverse_series(_scalar())
    for radius in (-1, certificate.parameter_radius, 2 * certificate.parameter_radius):
        with pytest.raises(ValueError):
            certificate.tail_bound(radius)
    for value in (True, 0.1):
        with pytest.raises(TypeError):
            certificate.enclose(value)
    with pytest.raises(ValueError):
        certificate.enclose(-certificate.parameter_radius)


def test_factory_only_and_immutable_certificate():
    with pytest.raises(TypeError):
        GaussianSeriesCertificate()
    with pytest.raises(TypeError):
        certify_gaussian_inverse_series(object())
    certificate = certify_gaussian_inverse_series(_scalar())
    with pytest.raises(FrozenInstanceError):
        certificate.parameter_radius = Fraction(100)


def test_certification_refuses_work_bit_and_finite_norm_failures():
    assert certify_gaussian_inverse_series(_scalar(1, budget=RidgeBudget(max_elimination_work=300))) is None
    assert certify_gaussian_inverse_series(_scalar(1, budget=RidgeBudget(max_coefficient_bits=64))) is None
    assert certify_gaussian_inverse_series(_scalar(), parameter_radius=2**5000) is None
    enormous = gaussian_inverse_series(((-1,), (1,)), (Fraction(1, 2),) * 2, (0,), ((0,), (10**400,)))
    assert certify_gaussian_inverse_series(enormous) is None


def test_evaluation_refuses_endpoint_overflow_without_infinite_certificate():
    series = gaussian_inverse_series(((0,),), (1,), (0,), ((0,), (1,)))
    certificate = certify_gaussian_inverse_series(series)
    with pytest.raises(ArithmeticError):
        certificate.enclose(10**400)


def test_real_contraction_does_not_claim_an_entire_nonlinear_inverse():
    mp = pytest.importorskip("mpmath")
    certificate = certify_gaussian_inverse_series(_scalar())
    with mp.workdps(80):
        # For U(x)=x-tanh(x/2), the branch from zero reaches U'(i*pi/2)=0.
        critical_x = mp.j * mp.pi / 2
        critical_target = critical_x - mp.tanh(critical_x / 2)
        derivative = 1 - mp.sech(critical_x / 2)**2 / 2
        assert abs(derivative) < mp.mpf("1e-70")
        assert 0 < _mp(mp, certificate.parameter_radius) < abs(critical_target)


@pytest.mark.parametrize("q", [Fraction(1, 2), Fraction(99, 100), Fraction(9999, 10000)])
def test_stronger_rational_strip_radius_improves_the_previous_ball_bound(q):
    # With centers±1 and nu1, time1/(2q) gives the requested exact q.
    time = 1/(2*q)
    series = _scalar(order=3, time=time)
    certificate = certify_gaussian_inverse_series(series)
    assert certificate is not None and series.contraction_bound == q
    old_spatial = 2*time*(1-q)/(4*2)
    old_kappa = q*(3+q)**2/(4*(1+q)**2)
    old_parameter = (1-old_kappa)*old_spatial/time
    assert certificate.spatial_radius > old_spatial
    assert certificate.complex_contraction_bound == 4*q/(3+q)
    assert certificate.parameter_radius > 7*old_parameter
    if q >= Fraction(99, 100):
        assert certificate.parameter_radius > 50*old_parameter
    # Replay the algebraic phase-width implication without transcendental values.
    phase_squared = series.diameter_squared * certificate.spatial_radius**2/(16*time**2)
    assert phase_squared <= (1-q)/4
    assert 0 <= certificate.complex_contraction_bound < 1


def test_two_center_complex_covariance_attains_the_uniform_strip_bound():
    mp = pytest.importorskip("mpmath")
    with mp.workdps(100):
        diameter, viscosity, time = mp.mpf(3), mp.mpf(2), mp.mpf("1.5")
        centers = (-diameter/2, diameter/2)
        q = diameter**2/(8*viscosity*time)
        for eta in (mp.mpf("0.1"), mp.mpf("0.4"), mp.mpf("0.9")):
            imaginary = 4*viscosity*time*eta/diameter
            for real in (mp.mpf(-5), mp.mpf(0), mp.mpf(7)):
                real_weights = [mp.exp(a*real/(2*viscosity*time)) for a in centers]
                real_weights = [weight/sum(real_weights) for weight in real_weights]
                phases = [a*imaginary/(2*viscosity*time) for a in centers]
                partition = sum(p*mp.exp(mp.j*theta) for p, theta in zip(real_weights, phases, strict=True))
                posterior = [p*mp.exp(mp.j*theta)/partition
                             for p, theta in zip(real_weights, phases, strict=True)]
                mean = sum(p*a for p, a in zip(posterior, centers, strict=True))
                covariance = sum(p*a*a for p, a in zip(posterior, centers, strict=True)) - mean*mean
                pair_covariance = sum(real_weights[j]*real_weights[k]*(centers[j]-centers[k])**2
                                      * mp.exp(mp.j*(phases[j]+phases[k])) for j in range(2) for k in range(2))
                pair_covariance /= 2*partition**2
                assert abs(covariance-pair_covariance) < mp.mpf("1e-90")
                bound = q/mp.cos(eta)**2
                assert abs(covariance/(2*viscosity*time)) <= bound + mp.mpf("1e-90")
                if real == 0:
                    assert abs(abs(partition)-mp.cos(eta)) < mp.mpf("1e-90")
                    assert abs(covariance/(2*viscosity*time)-bound) < mp.mpf("1e-90")


def test_complex_covariance_bound_has_no_extra_dimension_factor():
    mp = pytest.importorskip("mpmath")
    with mp.workdps(90):
        centers = (mp.matrix([0, 0]), mp.matrix([1, 0]), mp.matrix([0, 1]))
        diameter = mp.sqrt(2)
        for real in (mp.matrix([0, 0]), mp.matrix([4, -3]), mp.matrix([-10, 8])):
            imaginary = mp.matrix([mp.mpf("0.3"), mp.mpf("-0.4")])
            weights = [mp.exp((a.T*real)[0]/2) for a in centers]
            total = sum(weights)
            phases = [(a.T*imaginary)[0]/2 for a in centers]
            z = sum(p*mp.exp(mp.j*theta) for p, theta in zip(weights, phases, strict=True))/total
            posterior = [p*mp.exp(mp.j*theta)/(total*z) for p, theta in zip(weights, phases, strict=True)]
            mean = sum((p*a for p, a in zip(posterior, centers, strict=True)), mp.zeros(2, 1))
            covariance = sum((p*(a*a.T) for p, a in zip(posterior, centers, strict=True)), mp.zeros(2)) - mean*mean.T
            _, singular_values, _ = mp.svd(covariance)
            eta = diameter*mp.norm(imaginary)/4
            assert singular_values[0] <= diameter**2/(4*mp.cos(eta)**2)
