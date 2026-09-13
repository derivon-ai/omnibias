# SPDX-License-Identifier: Apache-2.0
"""Certified Hamiltonian line amplitudes from actual charged-energy bounds.

The amplitude is <Phi_gamma, exp[-T(A_charged-E0)] Phi_gamma>, where
Phi_gamma=psi0 U_gamma/sqrt(2) on a shortest path. It is not an assumed
isotropic Euclidean Wilson-measure expectation. The written spectral/Jensen
proof is in docs/api/gauge-charged-amplitude.md.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import certificate_mode, exp_iv
from omnibias.geometry.gauge.transfer.charged_confinement import (
    replay_su2_static_confinement_certificate,
)
from omnibias.geometry.gauge.transfer.static_sources import _rational


def su2_hamiltonian_rectangle_enclosure(
    confinement_certificate: dict[str, Any],
    *,
    time: int | Fraction = 1,
) -> dict[str, Any]:
    """Enclose the actual shortest-path Hamiltonian amplitude at exact T>=0.

    The input must be a passing explicit-graph confinement certificate with
    its complete Fourier parent. An asserted energy floor or family label is
    insufficient. Strict positivity of a rounded amplitude endpoint is not
    inferred when exponential underflow produces zero.
    """
    duration = _rational(time, "time")
    if duration < 0:
        raise ValueError("time must be nonnegative")
    if not isinstance(confinement_certificate, dict):
        raise TypeError("a complete explicit-graph confinement certificate is required")
    if not replay_su2_static_confinement_certificate(confinement_certificate):
        raise ValueError("the explicit-graph confinement certificate must pass replay")
    source = deepcopy(confinement_certificate)
    w = source["payload"]["witness"]
    # A quantified family does not identify one path or one amplitude.
    if "static_energy_enclosure" not in w or "graph_distance" not in w:
        raise ValueError("an explicit graph and its shortest path are required")
    lower_energy, upper_energy = map(Fraction, w["static_energy_enclosure"])
    with certificate_mode():
        if duration == 0:
            lower, upper = Fraction(1), Fraction(1)
        else:
            low = exp_iv(-Interval.from_value(duration * upper_energy))
            high = exp_iv(-Interval.from_value(duration * lower_energy))
            lower, upper = max(Fraction(0), Fraction(low.lo)), min(Fraction(1), Fraction(high.hi))
        witness = {
            "model": "su2_shortest_path_hamiltonian_amplitude_v1",
            "amplitude": "<Phi_gamma, exp[-T*(A_charged-E0)] Phi_gamma>",
            "state": "Phi_gamma=psi0*U_gamma/sqrt(2), with true normalized neutral vacuum psi0",
            "time": str(duration), "time_units": "dimensionless; conjugate to aH",
            "kappa": w["kappa"], "source": w["source"], "target": w["target"],
            "graph_distance": w["graph_distance"], "path": w["path"],
            "dimensionless_rectangle_area": str(duration * w["graph_distance"]),
            "spectral_support_lower": str(lower_energy),
            "exact_state_energy_mean": str(upper_energy),
            "lower_envelope_negative_exponent": str(duration * upper_energy),
            "upper_envelope_negative_exponent": str(duration * lower_energy),
            "amplitude_enclosure": [str(lower), str(upper)],
            "strictly_positive_numeric_lower": lower > 0,
            "rectangle_envelope_verified": True,
            "isotropic_euclidean_wilson_identification_verified": False,
            "asymptotic_string_tension_limit_verified": False,
            "continuum_claim": False, "yang_mills_claim": False,
        }
        certificate = make_certificate(
            claim="two-sided Hamiltonian line-amplitude enclosure from replayed charged spectral support",
            payload={"type": "su2_hamiltonian_rectangle_v1", "source_certificate": source,
                     "witness": witness},
            honesty={"unconditional_transcendentals": True,
                     "continuum_claim": False, "yang_mills_claim": False},
            meta={"analytic_implication": "docs/api/gauge-charged-amplitude.md",
                  "scope": "fixed graph, all spins, exact vacuum centering, dimensionless Hamiltonian time"},
        )
    return {
        "status": "PASS", "verification_kind": "CERTIFIED_INTERVAL_WITH_WRITTEN_SPECTRAL_IMPLICATION",
        "witness": witness, "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        "hamiltonian_rectangle_envelope_verified": True,
        "isotropic_euclidean_wilson_identification_verified": False,
        "asymptotic_string_tension_limit_verified": False,
        "theorem_prover_verified": False, "mathlib_verified": False,
        "analytic_implication_formally_verified": False,
        "continuum_claim": False, "yang_mills_claim": False,
    }


def replay_su2_hamiltonian_rectangle_certificate(certificate: dict[str, Any]) -> bool:
    """Recompute the entire nested proof and every envelope and scope field."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_hamiltonian_rectangle_v1":
            return False
        report = su2_hamiltonian_rectangle_enclosure(
            payload["source_certificate"], time=Fraction(payload["witness"]["time"]))
        return bool(report["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False
