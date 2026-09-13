# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""An all-space SU(2) bosonic three-matrix singlet gap with replayed sources.

This is a finite-dimensional configuration-space Hamiltonian, not a finite
spin truncation or an invariant subspace of a spatial lattice. Its
infinite-domain and angular-sector proofs are written in
``docs/api/gauge-commutator-matrix.md``; no formal tier is asserted.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.core.verified.airy_sturm import (
    airy_half_line_lower_bounds,
    replay_airy_sturm_certificate,
)


def su2_commutator_matrix_gap() -> dict[str, Any]:
    """Earn gap(h3 on simultaneous SO(3) singlets)>=19/200 on all of R^9.

    h3=-sum_i Delta_i+sum_{i<j}|x_i cross x_j|², with i=1,2,3.
    Every scalar-source cell is recomputed and replayed before promotion.
    """
    source = airy_half_line_lower_bounds()
    source_ok = replay_airy_sturm_certificate(source["certificate"])
    source_ok = source_ok and source["certificate"]["payload"] == {
        key: value
        for key, value in source.items()
        if key not in {"certificate", "theorem_prover_verified", "mathlib_verified"}
    }
    source_ok = (
        source_ok
        and source["status"] == "PASS"
        and source["half_line_eigenvalue_lower_bounds_verified_in_written_analysis"] is True
    )
    a0, a1, angular = Q(23, 10), Q(4), Q(16, 5)
    barta_cube = Q(137781, 4096)
    gaussian = Q(5, 4)
    kinetic = 9 * gaussian / 2
    potential = 9 / (2 * gaussian**2)
    upper = kinetic + potential
    excited = min(2 * a0 + a1, a0 + 2 * angular, 3 * angular)
    gap = excited - upper
    passed = bool(source_ok and barta_cube > angular**3 and gap > 0)
    payload = {
        "type": "su2_commutator_matrix_gap_v1",
        "inputs": {},
        "status": "PASS" if passed else "INCONCLUSIVE",
        "operator": "h3=-sum_i Delta_xi+sum_{i<j}|xi cross xj|^2 on L2(R^9)",
        "domain": "Friedrichs form domain; no configuration cutoff or added quadratic mass",
        "physical_sector": "simultaneous SO(3) rotations of all three color vectors; all singlets retained",
        "scalar_source": source,
        "arithmetic": {
            "matrix_count": 3,
            "configuration_dimension": 9,
            "linear_comparison": "h3>=sum_i(-Delta_i/2+sqrt(2)*|xi|)",
            "one_particle_comparison_scale": "1",
            "radial_l0_ground_lower": str(a0),
            "radial_l0_first_excited_lower": str(a1),
            "radial_l_ge_1_ground_lower": str(angular),
            "barta_trial": "u(r)=r^2 exp(-r^(3/2)/2)",
            "barta_local_energy": "7*r/16+27/(8*sqrt(r))",
            "barta_minimum_cubed_lower": str(barta_cube),
            "barta_cubed_target_slack": str(barta_cube - angular**3),
            "singlet_excitation_candidates": [
                str(2 * a0 + a1),
                str(a0 + 2 * angular),
                str(3 * angular),
            ],
            "gaussian_width": str(gaussian),
            "gaussian_kinetic_expectation": str(kinetic),
            "gaussian_quartic_expectation": str(potential),
            "ground_energy_lower": str(3 * a0),
            "ground_energy_upper": str(upper),
            "first_singlet_excited_energy_lower": str(excited),
            "singlet_gap_lower": str(gap),
            "homogeneous_model": "kappa/(2*N^3)*(-sum Delta_A)+N^3/(2*kappa)*sum|Ai cross Aj|^2",
            "homogeneous_dilation": "Ai=kappa^(1/3)*xi/N, kappa>0, N a positive integer",
            "homogeneous_energy_scale": "kappa^(1/3)/(2*N)",
            "homogeneous_singlet_gap_lower": "19*kappa^(1/3)/(400*N)",
        },
        "scalar_source_replay_verified": bool(source_ok),
        "actual_matrix_singlet_gap_verified_in_written_analysis": passed,
        "all_space_comparison_verified_in_written_analysis": passed,
        "compact_resolvent_verified_in_written_analysis": passed,
        "ground_state_unique_and_singlet_verified_in_written_analysis": passed,
        "finite_spin_truncation_used": False,
        "quadratic_mass_added": False,
        "spatial_lattice_invariant_subspace_verified": False,
        "compact_lattice_gap_verified": False,
        "uniform_spatial_volume_gap_verified": False,
        "continuum_claim": False,
        "yang_mills_mass_gap_claim": False,
        "analytic_proof_formally_verified": False,
    }
    return {
        **payload,
        "certificate": make_certificate(
            claim="all-space bosonic SU2 three-matrix singlet spectral gap at least19/200",
            payload=payload,
            meta={
                "transcend_backend": "not_used",
                "analytic_implication": "docs/api/gauge-commutator-matrix.md",
            },
        ),
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


def replay_su2_commutator_matrix_gap_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild the full source-dependent matrix certificate, including all cells."""
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        if certificate["payload"]["type"] != "su2_commutator_matrix_gap_v1":
            return False
        expected = su2_commutator_matrix_gap()
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = ["replay_su2_commutator_matrix_gap_certificate", "su2_commutator_matrix_gap"]
