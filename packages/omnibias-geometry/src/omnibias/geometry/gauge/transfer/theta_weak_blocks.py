# SPDX-License-Identifier: Apache-2.0
"""Actual weak-coupling theta gaps, conditional blocks and a correlation lower bound.

The all-angular reduced operator and its original-link conditional frames are
proved in docs/api/gauge-theta-weak-blocks.md. Canonical rational replay checks
the displayed source and budgets; it does not formalize the operator proof.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _rational

_KAPPA_MAX = Q(1, 64)
_GAP = Q(2, 5)


def _plaquette_trial_source(coupling: Q) -> tuple[dict[str, Any], bool, Q | None]:
    # Only the canonical source's sealed trial-energy upper bound is consumed.
    # Its already established spectral gap is not an input to this theorem.
    from omnibias.geometry.gauge.transfer.weak_plaquette import (
        replay_su2_weak_plaquette_gap_certificate,
        su2_weak_plaquette_gap,
    )

    result = su2_weak_plaquette_gap(coupling)
    certificate = result.get("certificate", {})
    if not isinstance(certificate, dict):
        return {}, False, None
    replayed = replay_su2_weak_plaquette_gap_certificate(certificate)
    try:
        witness = certificate["payload"]["witness"]
        ground = Q(witness["arithmetic"]["ground_energy_upper"])
        verified = bool(
            replayed
            and certificate["payload"]["type"] == "su2_weak_plaquette_gap_v1"
            and Q(witness["inputs"]["kappa"]) == coupling
            and witness["inputs"]["cutoff"] is None
            and certificate["honesty"]["actual_plaquette_gap_verified"] is True
            and 0 <= ground <= 3
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        ground, verified = None, False
    return certificate, verified, ground


def su2_theta_weak_block_gaps(kappa: int | Q = _KAPPA_MAX) -> dict[str, Any]:
    """Certify an actual two-plaquette gap for every 0<kappa<=1/64.

    The full reduced SU2^2 gap is at least 2/5. In the original seven-edge
    theta vacuum the two forest blocks have conditional quantum gaps 2/15
    and 4/25, retaining all boundary flux and imposing only internal Gauss
    constraints. The separately earned maximal-correlation *lower* bound
    max(0,1-3*kappa/4) holds for every positive coupling.

    Positive couplings outside the gap interval return INCONCLUSIVE; this
    does not assert absence of a gap. Inputs must be integers or Fractions.
    """
    coupling = _rational(kappa, "kappa")
    if coupling <= 0:
        raise ValueError("kappa must be strictly positive")
    source, source_verified, single_ground = _plaquette_trial_source(coupling)
    ground = 2 * single_ground if source_verified and single_ground is not None else None
    radial_branch = Q(779, 121) - 39 * coupling / 32
    angular_branch = Q(76, 11) - 33 * coupling / 16
    excited = min(radial_branch, angular_branch)
    gates = {
        "canonical_plaquette_trial_source": source_verified,
        "requested_kappa_in_proved_interval": coupling <= _KAPPA_MAX,
        "rational_sqrt3_lower": Q(19, 11) ** 2 <= 3,
        "radial_branch_above_32_5": radial_branch >= Q(32, 5),
        "angular_branch_above_32_5": angular_branch >= Q(32, 5),
        "actual_trial_ground_upper_at_most_6": ground is not None and ground <= 6,
    }
    passed = all(gates.values())
    correlation = max(Q(0), 1 - 3 * coupling / 4) if source_verified else None
    arithmetic = {
        "kappa": str(coupling),
        "proved_kappa_upper": str(_KAPPA_MAX),
        "sqrt3_rational_lower": "19/11",
        "pi_rational_upper": "22/7",
        "single_plaquette_ground_energy_upper": str(single_ground)
        if single_ground is not None
        else None,
        "ground_energy_upper": str(ground) if ground is not None else None,
        "radial_excitation_branch_lower": str(radial_branch),
        "angular_excitation_branch_lower": str(angular_branch),
        "first_reduced_excitation_lower": str(excited),
        "radial_branch_slack": str(radial_branch - Q(32, 5)),
        "angular_branch_slack": str(angular_branch - Q(32, 5)),
        "full_reduced_gap_lower": str(_GAP) if passed else None,
        "physical_theta_gap_lower": str(_GAP) if passed else None,
        "conditional_A_form_fraction": "1/3",
        "conditional_B_form_fraction": "2/5",
        "conditional_A_quantum_gap_lower": "2/15" if passed else None,
        "conditional_B_quantum_gap_lower": "4/25" if passed else None,
        "conditional_A_poincare_gap_lower": str(Q(4, 15) / coupling) if passed else None,
        "conditional_B_poincare_gap_lower": str(Q(8, 25) / coupling) if passed else None,
        "forest_maximal_correlation_lower": str(correlation) if correlation is not None else None,
        "correlation_lower_formula": "max(0,1-3*kappa/4)",
        "gates": gates,
    }
    scopes = {
        "actual_theta_vacuum_identified_in_written_analysis": source_verified,
        "actual_full_reduced_gap_verified_in_written_analysis": passed,
        "actual_physical_theta_gap_verified_in_written_analysis": passed,
        "actual_conditional_block_gaps_verified_in_written_analysis": passed,
        "all_boundary_flux_sectors_retained": passed,
        "all_reduced_representations_included": passed,
        "actual_forest_correlation_lower_verified_in_written_analysis": source_verified,
        "correlation_lower_is_overlap_upper": False,
        "physical_projection_angle_lower_verified": False,
        "source_gap_used_as_premise": False,
        "unrestricted_original_seven_link_gap_verified": False,
        "ambient_exterior_uniformity_verified": False,
        "uniform_in_volume_claim": False,
        "all_scale_refinement_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
        "analytic_proof_formally_verified": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }
    payload = {
        "type": "su2_theta_weak_block_gaps_v1",
        "status": "PASS" if passed else "INCONCLUSIVE",
        "inputs": {"kappa": str(coupling)},
        "plaquette_trial_source_certificate": source,
        "model": "seven original unit-weight edges, two adjacent square plaquettes",
        "original_edges": [[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5]],
        "blocks": {
            "A_edges": [1, 2, 6],
            "B_edges": [3, 4, 5, 7],
            "internal_A_vertices": [1],
            "internal_B_vertices": [3, 5],
            "boundary_vertices": [0, 2, 4],
        },
        "normalization": "H=kappa*C/2+2*(A(x)+A(y))/kappa; A(U)=2-Tr(U), C_fund=3/4",
        "coordinates": "a=e1^-1,b=e2,s=e6,P=e5*e3,Q=e7*e4^-1; x=s^-1*a*P,y=s^-1*b*Q",
        "reduced_electric_operator": "C=3*(Cx+Cy)+C_simultaneous_left",
        "conditional_A_operator": "CA=Cx+Cy+C_simultaneous_left >= C/3",
        "conditional_B_operator": "CB=2*(Cx+Cy) >= 2*C/5",
        "gap_sector": "full reduced L2(SU2^2); consequently simultaneous-conjugation invariants",
        "conditional_sector": "internal-Gauss invariants, arbitrary boundary representations",
        "conditional_density": "the same actual normalized Phi(x,y)^2 dHaar(x)dHaar(y) in both frozen fibers",
        "vacuum_subtraction": "unique positive actual theta ground state; not isolated conditional Hamiltonians",
        "correlation_scope": "maximal correlation on unrestricted L2 spaces of the two forest marginals in the actual theta vacuum, all positive kappa",
        "arithmetic": arithmetic,
        **scopes,
    }
    certificate = make_certificate(
        claim="actual weak-coupling theta spectral and conditional block gaps, with a separate forest correlation lower bound",
        payload=payload,
        meta={
            "analytic_implication": "docs/api/gauge-theta-weak-blocks.md",
            "transcend_backend": "not_used",
            "proof_register": "written Barta/minmax/conditional-frame proof and exact rational canonical replay",
        },
    )
    return {**payload, "certificate": certificate}


def replay_su2_theta_weak_block_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild both decisions, nested trial evidence, bounds and sector scope."""
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_theta_weak_block_gaps_v1":
            return False
        expected = su2_theta_weak_block_gaps(Q(payload["inputs"]["kappa"]))
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = [
    "replay_su2_theta_weak_block_certificate",
    "su2_theta_weak_block_gaps",
]
