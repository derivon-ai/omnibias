# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact pure-electric oracles and proof-envelope/replay regressions."""
import random
from copy import deepcopy
from fractions import Fraction as Q

import mpmath
import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.charged_amplitude import (
    replay_su2_hamiltonian_rectangle_certificate,
    su2_hamiltonian_rectangle_enclosure,
)
from omnibias.geometry.gauge.transfer.charged_confinement import (
    su2_static_confinement_bounds,
    su2_static_confinement_family,
)
from omnibias.geometry.gauge.transfer.vacuum_fourier import (
    su2_vacuum_fourier_bounds,
    su2_vacuum_fourier_family,
)


def _source(distance=2, coupling=8):
    fourier = su2_vacuum_fourier_bounds(
        distance + 1, [(i, i + 1) for i in range(distance)], kappa=coupling)
    return su2_static_confinement_bounds(fourier["certificate"], 0, distance)["certificate"]


def test_zero_time_is_exact_and_replays():
    report = su2_hamiltonian_rectangle_enclosure(_source(), time=0)
    assert report["witness"]["amplitude_enclosure"] == ["1", "1"]
    assert replay_su2_hamiltonian_rectangle_certificate(report["certificate"])


def test_true_pure_electric_amplitude_dense_grid_and_random_samples():
    rng = random.Random(281)
    times = [Q(i, 20) for i in range(41)]
    times += [Q(rng.randrange(0, 1001), 137) for _ in range(40)]
    source = _source(distance=3, coupling=4)
    with mpmath.workdps(90):
        for duration in times:
            report = su2_hamiltonian_rectangle_enclosure(source, time=duration)
            pair = list(map(Q, report["witness"]["amplitude_enclosure"]))
            t = mpmath.mpf(duration.numerator) / duration.denominator
            # The pure-electric path is an EXACT all-spin eigenfunction,
            # so its spectral measure is a point mass at 3*kappa*d/8.
            true = mpmath.exp(-mpmath.mpf(9) * t / 2)
            lo = mpmath.mpf(pair[0].numerator) / pair[0].denominator
            hi = mpmath.mpf(pair[1].numerator) / pair[1].denominator
            assert lo <= true <= hi


def test_interacting_envelopes_have_correct_area_and_energy_conventions():
    fourier = su2_vacuum_fourier_bounds(
        4, [(0, 1), (1, 2), (2, 3), (3, 0)],
        plaquettes=[(1, 2, 3, 4)], kappa=64)
    source = su2_static_confinement_bounds(fourier["certificate"], 0, 2)["certificate"]
    report = su2_hamiltonian_rectangle_enclosure(source, time=Q(1, 4))
    w = report["witness"]
    assert w["dimensionless_rectangle_area"] == "1/2"
    assert w["spectral_support_lower"] == "28"
    assert w["exact_state_energy_mean"] == "48"
    assert w["lower_envelope_negative_exponent"] == "12"
    assert w["upper_envelope_negative_exponent"] == "7"
    assert replay_su2_hamiltonian_rectangle_certificate(report["certificate"])
    for name in ("continuum_claim", "yang_mills_claim", "theorem_prover_verified",
                 "isotropic_euclidean_wilson_identification_verified",
                 "asymptotic_string_tension_limit_verified"):
        assert report[name] is False


def test_underflow_remains_sound_without_false_positive_endpoint():
    report = su2_hamiltonian_rectangle_enclosure(_source(), time=10000)
    lo, hi = map(Q, report["witness"]["amplitude_enclosure"])
    assert lo == 0 and hi > 0
    assert not report["witness"]["strictly_positive_numeric_lower"]
    assert replay_su2_hamiltonian_rectangle_certificate(report["certificate"])


def test_family_certificate_does_not_identify_one_rectangle():
    family = su2_static_confinement_family(su2_vacuum_fourier_family(64, 4)["certificate"])
    with pytest.raises(ValueError, match="explicit graph"):
        su2_hamiltonian_rectangle_enclosure(family["certificate"])


@pytest.mark.parametrize("field,value", [
    ("exact_state_energy_mean", "1"), ("spectral_support_lower", "1000"),
    ("time_units", "physical seconds"), ("continuum_claim", True),
    ("isotropic_euclidean_wilson_identification_verified", True),
    ("amplitude_enclosure", ["0", "0"]),
])
def test_rehashed_claims_are_refused(field, value):
    c = deepcopy(su2_hamiltonian_rectangle_enclosure(_source())["certificate"])
    c["payload"]["witness"][field] = value
    c.pop("digest")
    assert not replay_su2_hamiltonian_rectangle_certificate(seal_certificate(c))


@pytest.mark.parametrize("duration", [-1, True, 1.0])
def test_inexact_or_negative_time_is_refused(duration):
    with pytest.raises((ValueError, TypeError)):
        su2_hamiltonian_rectangle_enclosure(_source(), time=duration)


@pytest.mark.parametrize("value", [None, [], {}, {"payload": []}])
def test_malformed_replay_returns_false(value):
    assert not replay_su2_hamiltonian_rectangle_certificate(value)
