# SPDX-License-Identifier: Apache-2.0
"""Small-coupling SU(2) three-loop gap in the center-even physical sector.

The compact model and all seven other center sectors remain distinct. The
analytic proof is docs/api/gauge-compact-commutator.md; exact arithmetic and
canonical matrix-source replay do not formalize that proof.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _rational

_KAPPA_MAX = Q(1, 2**36)
_TAU_MAX = Q(1, 64)
_X_MAX = _TAU_MAX**2
_GROUND_CAP = Q(1701, 200)
_EXCITED_FLOOR = Q(43, 5)
_ROOT_BITS = 80


def _integer_cube_root(value: int) -> int:
    """Floor of the nonnegative integer cube root, without floating arithmetic."""
    if type(value) is not int or value < 0:
        raise ValueError("a nonnegative exact integer is required")
    if value == 0:
        return 0
    lo, hi = 0, 1 << ((value.bit_length() + 2) // 3)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid**3 <= value:
            lo = mid
        else:
            hi = mid
    return lo


def _cube_root_enclosure(value: Q) -> tuple[Q, Q]:
    """A positive relative dyadic enclosure, including arbitrarily small inputs."""
    if value <= 0:
        raise ValueError("cube-root input must be positive")
    binary_exponent = value.numerator.bit_length() - value.denominator.bit_length()
    power = Q(2**binary_exponent) if binary_exponent >= 0 else Q(1, 2 ** (-binary_exponent))
    if value < power:
        binary_exponent -= 1
    root_exponent = binary_exponent // 3
    scale = Q(2**root_exponent) if root_exponent >= 0 else Q(1, 2 ** (-root_exponent))
    normalized = value / scale**3
    numerator = normalized.numerator << (3 * _ROOT_BITS)
    denominator = normalized.denominator
    integer = _integer_cube_root(numerator // denominator)
    lo = scale * Q(integer, 2**_ROOT_BITS)
    if integer**3 * denominator == numerator:
        return lo, lo
    return lo, scale * Q(integer + 1, 2**_ROOT_BITS)


def _matrix_source() -> tuple[dict[str, Any], bool, Q | None, Q | None]:
    # Dynamic import keeps the consumer attached to the current canonical
    # scalar/Sturm source rather than treating copied spectrum flags as inputs.
    from omnibias.geometry.gauge.transfer.commutator_matrix import (
        replay_su2_commutator_matrix_gap_certificate,
        su2_commutator_matrix_gap,
    )

    result = su2_commutator_matrix_gap()
    certificate = result["certificate"]
    replayed = replay_su2_commutator_matrix_gap_certificate(certificate)
    payload = certificate.get("payload", {})
    arithmetic = payload.get("arithmetic", {})
    try:
        ground = Q(arithmetic["ground_energy_upper"])
        excited = Q(arithmetic["first_singlet_excited_energy_lower"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        ground, excited = None, None
    verified = (
        replayed
        and payload.get("status") == "PASS"
        and payload.get("actual_matrix_singlet_gap_verified_in_written_analysis") is True
        and payload.get("all_space_comparison_verified_in_written_analysis") is True
        and payload.get("compact_resolvent_verified_in_written_analysis") is True
        and payload.get("ground_state_unique_and_singlet_verified_in_written_analysis") is True
        and ground is not None
        and excited is not None
        and ground <= _GROUND_CAP
        and excited >= _EXCITED_FLOOR
    )
    return certificate, verified, ground, excited


def su2_compact_commutator_gap(kappa: int | Q = _KAPPA_MAX) -> dict[str, Any]:
    """Certify the actual center-even three-selfloop gap for 0<kappa<=2**(-36).

    The Hilbert space consists of simultaneous-Ad-invariant functions on
    SU(2)^3 that are even under each of the three independent center flips.
    The gap is at least kappa**(1/3)/50. The seven other center characters
    instead receive upper bounds 91*kappa**(2/3) on their ground-energy
    differences from the even vacuum. Neither result is a bulk theorem.

    Integer and Fraction inputs are accepted, floats and booleans rejected.
    Positive inputs outside the written interval yield INCONCLUSIVE. The
    numerical gap floor uses a normalized, exact 80-bit cube-root enclosure;
    it remains positive even for inputs below an absolute dyadic grid.
    """
    coupling = _rational(kappa, "kappa")
    if coupling <= 0:
        raise ValueError("kappa must be strictly positive")
    source, source_verified, ground, excited = _matrix_source()
    root_lo, root_hi = _cube_root_enclosure(coupling)
    x = _X_MAX
    f = (1 - 4 * x) ** 3
    haar_upper = 1 / (1 - 4 * x) ** 2
    norm_lower = 1 - 9 * x / 10
    normalized_lower = _EXCITED_FLOOR * f - 5 * x / 8
    normalized_upper = haar_upper * (_GROUND_CAP + 109 * x / 16) / norm_lower
    normalized_gap = normalized_lower - normalized_upper
    flux_norm_lower = 1 - _GROUND_CAP * _TAU_MAX / 2
    flux_coefficient = (21 * _GROUND_CAP + Q(25, 8)) / 2
    bad_ratio_lower = 1 - 3 * _TAU_MAX**5 / 4
    bad_required_ratio = _EXCITED_FLOOR * _TAU_MAX / 2
    gates = {
        "canonical_matrix_source": source_verified,
        "requested_kappa_in_proved_interval": coupling <= _KAPPA_MAX,
        "positive_chart_and_cutoff_norm": 4 * x < 1 and norm_lower > 0,
        "bad_branch_above_compared_excitation": bad_ratio_lower >= bad_required_ratio,
        "normalized_gap_at_least_1_25": normalized_gap >= Q(1, 25),
        "flux_trial_norm_at_least_1_2": flux_norm_lower >= Q(1, 2),
        "haar_factor_at_most_two": haar_upper <= 2,
        "haar_excess_at_most_9x": haar_upper - 1 <= 9 * x,
        "good_factor_loss_at_most_12x": 1 - f <= 12 * x,
        "flux_coefficient_at_most_91": flux_coefficient <= 91,
        "cube_root_enclosure": 0 < root_lo <= root_hi and root_lo**3 <= coupling <= root_hi**3,
    }
    passed = all(gates.values())
    gap_lower = root_lo / 50 if passed else None
    flux_upper = 91 * root_hi**2 if passed else None
    arithmetic = {
        "kappa": str(coupling),
        "proved_kappa_upper": str(_KAPPA_MAX),
        "endpoint_tau": str(_TAU_MAX),
        "endpoint_x": str(x),
        "matrix_ground_energy_upper": str(ground) if ground is not None else None,
        "matrix_first_singlet_excited_energy_lower": str(excited) if excited is not None else None,
        "matrix_ground_cap_used": str(_GROUND_CAP),
        "matrix_first_excitation_floor_used": str(_EXCITED_FLOOR),
        "global_quantum_coercivity": "H>=kappa/4*sum C_i+sqrt(2)*sum|v_i|-3*kappa/4",
        "good_harmonic_factor_lower": str(f),
        "haar_density_factor_upper": str(haar_upper),
        "gaussian_cutoff_norm_lower": str(norm_lower),
        "normalized_first_excitation_lower": str(normalized_lower),
        "normalized_ground_energy_upper": str(normalized_upper),
        "normalized_gap_lower": str(normalized_gap),
        "normalized_gap_target": "1/25",
        "normalized_gap_slack": str(normalized_gap - Q(1, 25)),
        "bad_branch_ratio_lower": str(bad_ratio_lower),
        "bad_branch_required_ratio": str(bad_required_ratio),
        "flux_groundstate_cutoff_norm_lower": str(flux_norm_lower),
        "flux_coefficient_budget_upper": str(flux_coefficient),
        "cube_root_bits": _ROOT_BITS,
        "kappa_one_third_enclosure": [str(root_lo), str(root_hi)],
        "center_even_gap_coefficient": "1/50" if passed else None,
        "center_even_gap_lower": str(gap_lower) if gap_lower is not None else None,
        "other_center_sector_energy_difference_strict_lower": "0" if passed else None,
        "other_center_sector_energy_difference_upper": str(flux_upper)
        if flux_upper is not None
        else None,
        "other_center_sector_energy_difference_coefficient": "91" if passed else None,
        "other_center_sector_count": 7,
        "gates": gates,
    }
    scopes = {
        "actual_compact_center_even_gap_verified_in_written_analysis": passed,
        "actual_seven_flux_sector_upper_bounds_verified_in_written_analysis": passed,
        "all_spins_in_declared_compact_sector_verified": passed,
        "uniform_on_declared_small_coupling_interval_verified": passed,
        "unrestricted_gap_verified": False,
        "unrestricted_kappa_one_third_gap_verified": False,
        "exponential_tunneling_rate_verified": False,
        "arbitrary_exterior_conditional_verified": False,
        "uniform_in_volume_claim": False,
        "all_scale_refinement_claim": False,
        "continuum_claim": False,
        "yang_mills_mass_gap_claim": False,
        "analytic_proof_formally_verified": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }
    payload = {
        "type": "su2_compact_commutator_gap_v1",
        "status": "PASS" if passed else "INCONCLUSIVE",
        "inputs": {"kappa": str(coupling)},
        "matrix_source_certificate": source,
        "model": "one vertex, three SU2 selfloops, three pair commutator plaquettes",
        "normalization": "H=kappa/2*sum_i C_i+2/kappa*sum_(i<j)(2-Tr(U_i U_j U_i^-1 U_j^-1)); C_fund=3/4",
        "gap_sector": "simultaneous Ad invariants, even under each independent U_i -> -U_i",
        "other_sectors": "seven nontrivial Z2^3 characters, still simultaneous Ad invariants",
        "vacuum_subtraction": "the unique positive compact groundstate, which is Ad invariant and center even",
        "matrix_scaling": "v_i=(kappa^(1/3)/2)*Y_i; Euclidean reference=(kappa^(1/3)/2)*h3",
        "arithmetic": arithmetic,
        **scopes,
    }
    certificate = make_certificate(
        claim="actual center-even compact SU2 commutator gap and separate seven center-sector upper bounds",
        payload=payload,
        meta={
            "analytic_implication": "docs/api/gauge-compact-commutator.md",
            "transcend_backend": "not_used",
            "proof_register": "written compact all-state coercivity, sector-sensitive IMS/minmax, exact rational replay",
            "no_homogeneous_subspace_of_larger_lattice_assumed": True,
        },
    )
    return {**payload, "certificate": certificate}


def replay_compact_commutator_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild the complete source, requested coupling, sectors and both outcomes."""
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_compact_commutator_gap_v1":
            return False
        expected = su2_compact_commutator_gap(Q(payload["inputs"]["kappa"]))
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False


__all__ = [
    "replay_compact_commutator_certificate",
    "su2_compact_commutator_gap",
]
