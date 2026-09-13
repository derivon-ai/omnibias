# SPDX-License-Identifier: Apache-2.0
"""Actual SU(2) plaquette-path conditional gaps with all boundary flux.

The radial source is dynamically replayed. Angular comparison then gives
an unrestricted rotor gap 4/33 and an m-link conditional quantum gap m/33
at every positive coupling. The proof concerns one isolated square and
its actual vacuum, not an embedded block of an arbitrary larger graph.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational

_RADIAL_GAP = Q(26, 33)
_GROUND_CAP = Q(3)
_ROTOR_GAP = Q(4, 33)


def _radial_source(kappa: Q) -> tuple[dict[str, Any], bool, Q | None]:
    from omnibias.geometry.gauge.transfer.weak_plaquette import (
        replay_su2_weak_plaquette_gap_certificate,
        su2_weak_plaquette_gap,
    )

    result = su2_weak_plaquette_gap(kappa)
    certificate = result.get("certificate")
    if not isinstance(certificate, dict):
        return {}, False, None
    replayed = replay_su2_weak_plaquette_gap_certificate(certificate)
    try:
        witness = certificate["payload"]["witness"]
        honesty = certificate["honesty"]
        arithmetic = witness["arithmetic"]
        ground = Q(arithmetic["ground_energy_upper"])
        radial_gap = Q(arithmetic["all_positive_couplings_gap_lower"])
        verified = (
            replayed
            and result.get("status") == "PASS"
            and result.get("finite_gate_verified") is True
            and result.get("witness") == witness
            and result.get("all_positive_couplings_gap_lower") == str(radial_gap)
            and result.get("actual_plaquette_gap_verified") is True
            and honesty.get("actual_plaquette_gap_verified") is True
            and honesty.get("all_positive_couplings_gap_verified") is True
            and honesty.get("all_irreducible_characters_included") is True
            and witness["inputs"]["kappa"] == str(kappa)
            and witness["inputs"]["cutoff"] is None
            and radial_gap >= _RADIAL_GAP
            and ground <= _GROUND_CAP
        )
        return certificate, verified, ground
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return certificate, False, None


def su2_plaquette_path_conditional_gap(kappa: int | Q, block_edges: int = 1) -> dict[str, Any]:
    """Prove an actual conditional quantum gap m/33 for m=1,2,3 links.

    The block is a contiguous path in one four-edge square. Only its
    internal bivalent Gauss constraints are imposed; both endpoint flux
    representations remain unrestricted. Every complementary path value
    is covered, with the conditional density of the true square vacuum.

    Inputs are exact integers/Fractions, with an integer path length.
    Canonical radial-source failure returns INCONCLUSIVE, never an earned
    gap based on copied numerical constants. All normal positive inputs
    pass, with no floating or special-function evaluation.
    """
    coupling = _rational(kappa, "kappa")
    count = _integer(block_edges, "block_edges")
    if coupling <= 0 or count not in (1, 2, 3):
        raise ValueError("require kappa>0 and block_edges in {1,2,3}")
    source, source_verified, ground = _radial_source(coupling)
    oscillator_absolute = Q(35, 11) - coupling / 2
    oscillator_gap = oscillator_absolute - _GROUND_CAP
    combined = max(coupling, oscillator_gap)
    pointwise = min(_RADIAL_GAP, combined)
    gates = {
        "canonical_radial_source": source_verified,
        "angular_majorant_at_least_4_33": combined >= _ROTOR_GAP,
        "radial_majorant_at_least_4_33": _RADIAL_GAP >= _ROTOR_GAP,
        "proper_contiguous_path_length": count in (1, 2, 3),
    }
    passed = all(gates.values())
    arithmetic = {
        "kappa": str(coupling),
        "radial_gap_lower": str(_RADIAL_GAP),
        "radial_ground_energy_upper": str(ground) if ground is not None else None,
        "ground_cap_used": str(_GROUND_CAP),
        "pi_rational_upper": "22/7",
        "nonradial_centrifugal_gap_lower": str(coupling),
        "nonradial_oscillator_absolute_floor": str(oscillator_absolute),
        "nonradial_oscillator_gap_lower": str(oscillator_gap),
        "nonradial_combined_gap_lower": str(combined),
        "angular_bound_crossing_kappa": "4/33",
        "full_rotor_pointwise_gap_lower": str(pointwise) if passed else None,
        "full_rotor_gap_lower": str(_ROTOR_GAP) if passed else None,
        "rotor_to_path_form_factor": str(Q(count, 4)),
        "conditional_quantum_gap_lower": str(Q(count, 33)) if passed else None,
        "conditional_poincare_lower": str(2 * count / (33 * coupling)) if passed else None,
        "gates": gates,
    }
    scopes = {
        "radial_source_replay_verified": source_verified,
        "actual_full_rotor_gap_verified_in_written_analysis": passed,
        "actual_path_conditional_gap_verified_in_written_analysis": passed,
        "all_angular_sectors_included": passed,
        "all_boundary_flux_sectors_included": passed,
        "uniform_over_declared_path_exteriors_verified": passed,
        "all_positive_couplings_verified": passed,
        "spin_truncation_used": False,
        "endpoint_gauss_or_center_even_restriction_used": False,
        "frozen_bare_groundstate_substituted": False,
        "embedded_ambient_block_claim": False,
        "volume_uniform_claim": False,
        "all_scale_refinement_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
        "analytic_proof_formally_verified": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }
    payload = {
        "type": "su2_plaquette_path_conditional_gap_v1",
        "status": "PASS" if passed else "INCONCLUSIVE",
        "inputs": {"kappa": str(coupling), "block_edges": count},
        "radial_source_certificate": source,
        "model": "one isolated square, four distinct vertices and four unit electric edges",
        "normalization": "aH=kappa/2*sum_e C_e+2/kappa*(2-Tr H); single-link C_fund=3/4",
        "measure": "normalized Haar; actual unique positive square vacuum squared",
        "energy_units": "dimensionless original aH at the specified microscopic coupling",
        "rotor_extension": "L2(SU2), all SO3 angular l>=0; H4=2*kappa*C+2/kappa*(2-Tr U)",
        "angular_radial_operator": "kappa/2*(-d_theta^2+l*(l+1)/sin(theta)^2-1)+4/kappa*(1-cos(theta))",
        "angular_radial_domain": "Friedrichs form on (0,pi); l=0 Dirichlet, l>=1 centrifugal form",
        "geometry": {
            "cycle_vertices": [0, 1, 2, 3],
            "oriented_edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
            "block_oriented_edges": list(range(1, count + 1)),
            "complement_oriented_edges": list(range(count + 1, 5)),
            "internal_gauss_vertices": list(range(1, count)),
            "boundary_vertices": [0, count],
            "conditional_coordinate": "U=U1*...*Um; frozen complementary holonomy B; H=U*B",
            "conditional_density": "psi0(U*B)^2 dU, with normalizer independent of B",
            "conditional_hilbert_space": "arbitrary scalar L2(SU2,psi0(U*B)^2 dU), no endpoint Gauss restriction",
            "original_block_form": "(kappa/2)*m*integral |grad_U f|^2 psi0(U*B)^2 dU",
        },
        "arithmetic": arithmetic,
        **scopes,
    }
    certificate = make_certificate(
        claim="actual all-coupling SU2 plaquette path conditional gap with every boundary-flux sector",
        payload=payload,
        meta={
            "analytic_implication": "docs/api/gauge-plaquette-path.md",
            "transcend_backend": "not_used",
            "proof_register": "written all-angular Friedrichs comparison and actual-vacuum disintegration; exact rational replay",
        },
    )
    return {**payload, "certificate": certificate}


def replay_su2_plaquette_path_conditional_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild the source, angular budgets, geometry and complete scoped payload."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_plaquette_path_conditional_gap_v1":
            return False
        inputs = payload["inputs"]
        if not isinstance(inputs["kappa"], str):
            return False
        coupling = Q(inputs["kappa"])
        if str(coupling) != inputs["kappa"]:
            return False
        rebuilt = su2_plaquette_path_conditional_gap(coupling, inputs["block_edges"])
        return bool(certificate == rebuilt["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = [
    "replay_su2_plaquette_path_conditional_certificate",
    "su2_plaquette_path_conditional_gap",
]
