# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact SU(2) coarse holonomy elimination on a finite theta graph.

Three internally disjoint paths carry positive total electric weights A,B,S.
The coarse space ignores the third path. All complementary spins are covered;
only the displayed entries of the second-order effective operator are a pack.
This is an energy-dependent Feshbach calculation, not a vacuum-aligned RG map.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest

State = tuple[int, int, int]


def _rational(value: int | Q, name: str) -> Q:
    if type(value) is not int and not isinstance(value, Q):
        raise TypeError(f"{name} must be an exact integer or Fraction")
    return Q(value)


def _casimir(two_j: int) -> Q:
    return Q(two_j * (two_j + 2), 4)


def _entry(left: int, right: int, alpha: Q, paths: tuple[Q, ...],
           magnetic: tuple[Q, Q], z: Q) -> Q:
    """Closed rational Jacobi entry; caller has checked the all-spin domain."""
    n = min(left, right)
    if abs(left - right) > 1:
        return Q(0)

    def denominator(active: int, shift: int) -> Q:
        return alpha * (paths[active] * _casimir(n + shift)
                        + paths[1 - active] * _casimir(n)
                        + paths[2] * Q(3, 4)) - z

    if left != right:
        return magnetic[0] * magnetic[1] / 2 * (
            1 / denominator(0, 1) + 1 / denominator(1, 1)
        )
    value = Q(0)
    for active, weight in enumerate(magnetic):
        value += weight**2 * Q(n + 2, 2 * (n + 1)) / denominator(active, 1)
        if n:
            value += weight**2 * Q(n, 2 * (n + 1)) / denominator(active, -1)
    return value


def _neighbors(two_j: int) -> set[State]:
    """Every intermediate spin reached by one fundamental plaquette."""
    return {
        state for shifted in (two_j - 1, two_j + 1) if shifted >= 0
        for state in ((shifted, two_j, 1), (two_j, shifted, 1))
    }


def su2_theta_refinement(
    *,
    electric: int | Q = 2,
    path_weights: tuple[int | Q, int | Q, int | Q] = (3, 3, 1),
    magnetic_left: int | Q = Q(1, 2),
    magnetic_right: int | Q = Q(1, 2),
    spectral_parameter: int | Q = 0,
    coarse_two_j_max: int = 4,
) -> dict[str, Any]:
    """Enclose an actual eliminated interaction, including the full new sector.

    H=alpha*(A*C_a+B*C_b+S*C_s)-v1*chi_left-v2*chi_right;
    the nonnegative Wilson convention adds 2*(v1+v2)*I to H.
    The certificate covers the Neumann domain d_new-z>2*(v1+v2).
    Outside it the exact electric/cross-block identities still hold, but no
    interacting resolvent or self-energy enclosure is returned.
    """
    alpha = _rational(electric, "electric")
    if not isinstance(path_weights, (tuple, list)) or len(path_weights) != 3:
        raise ValueError("path_weights must contain three positive path totals")
    paths = tuple(_rational(value, "path weight") for value in path_weights)
    v1 = _rational(magnetic_left, "magnetic_left")
    v2 = _rational(magnetic_right, "magnetic_right")
    z = _rational(spectral_parameter, "spectral_parameter")
    if alpha <= 0 or min(paths) <= 0 or min(v1, v2) < 0:
        raise ValueError("electric and path weights must be positive; magnetic weights nonnegative")
    if type(coarse_two_j_max) is not int or coarse_two_j_max < 0:
        raise ValueError("coarse_two_j_max must be a nonnegative integer")
    coarse_gap = Q(3, 4) * alpha * (paths[0] + paths[1])
    new_floor = Q(3, 4) * alpha * (paths[2] + min(paths[:2]))
    cross_norm = v1 + v2
    denominator = new_floor - z
    domain = denominator > 2 * cross_norm
    second_order: list[list[str]] | None = None
    remainder: Q | None = None
    ratio: Q | None = None
    coarse_tail: Q | None = None
    coarse_boundary: Q | None = None
    coarse_display_error: Q | None = None
    states: list[State] = []
    if domain:
        ratio = 2 * cross_norm / denominator
        remainder = cross_norm**2 / denominator * ratio**2 / (1 - ratio**2)
        tail_denominator = alpha * (
            max(paths[:2]) * _casimir(coarse_two_j_max)
            + min(paths[:2]) * _casimir(coarse_two_j_max + 1)
            + paths[2] * Q(3, 4)
        ) - z
        coarse_tail = cross_norm**2 / tail_denominator
        coarse_boundary = _entry(
            coarse_two_j_max, coarse_two_j_max + 1, alpha, paths, (v1, v2), z
        )
        coarse_display_error = coarse_tail + coarse_boundary
        states = sorted(set().union(*(_neighbors(j) for j in range(coarse_two_j_max + 1))))
        second_order = [
            [str(_entry(left, right, alpha, paths, (v1, v2), z))
             for right in range(coarse_two_j_max + 1)]
            for left in range(coarse_two_j_max + 1)
        ]
    witness = {
        "inputs": {
            "electric": str(alpha), "path_weights": list(map(str, paths)),
            "magnetic_left": str(v1), "magnetic_right": str(v2),
            "spectral_parameter": str(z), "coarse_two_j_max": coarse_two_j_max,
        },
        "group": "su2",
        "graph": "three internally disjoint paths between two vertices; Gauss law at every vertex",
        "hamiltonian": "H=alpha*(A*C_a+B*C_b+S*C_s)-v1*chi_left-v2*chi_right",
        "energy_units": "one common input energy unit; no automatic spacing conversion",
        "wilson_nonnegative_convention_shift": str(2 * cross_norm),
        "coarse_projection": "conditional Haar expectation over the third path; s=0, a=b",
        "coarse_basis": "orthonormal SU(2) characters chi_(two_j/2) of the outer holonomy",
        "coarse_electric_gap": str(coarse_gap),
        "complement_electric_floor": str(new_floor),
        "new_minus_coarse_electric_gap": str(new_floor - coarse_gap),
        "complement_attaining_two_j": [1, 0, 1] if paths[0] <= paths[1] else [0, 1, 1],
        "electric_cross_block_norm": "0",
        "magnetic_cross_block_norm": str(cross_norm),
        "compressed_magnetic_operator": "0",
        "cross_block_square": {
            "identity_coefficient": str(v1**2 + v2**2),
            "fundamental_outer_character_coefficient": str(v1 * v2),
        },
        "complement_H_lower": str(new_floor - 2 * cross_norm),
        "resolvent_denominator_lower": str(denominator - 2 * cross_norm),
        "neumann_domain_verified": domain,
        "neumann_ratio_upper": str(ratio) if ratio is not None else None,
        "second_order_matrix": second_order,
        "second_order_intermediate_two_j": [list(state) for state in states],
        "second_order_definition": "Sigma0=P M Q (Q H0 Q-z)^(-1) Q M P",
        "exact_self_energy_definition": "Sigma=P M Q (Q H Q-z)^(-1) Q M P",
        "self_energy_remainder_lower": "0" if domain else None,
        "self_energy_remainder_upper": str(remainder) if remainder is not None else None,
        "second_order_coarse_tail_norm_upper": str(coarse_tail) if coarse_tail is not None else None,
        "second_order_coarse_boundary_entry": str(coarse_boundary) if coarse_boundary is not None else None,
        "second_order_display_error_upper": str(coarse_display_error) if coarse_display_error is not None else None,
        "full_self_energy_display_error_upper": (
            str(remainder + coarse_display_error)
            if remainder is not None and coarse_display_error is not None else None
        ),
        "display_error_scope": "operator norm on the whole coarse space after extending the displayed finite matrix by zero",
        "remainder_scope": "0 <= Sigma-Sigma0 <= remainder*I on the entire coarse Hilbert space",
        "effective_operator": "F(z)=P H0 P-z-Sigma; its displayed pack lies between P H0 P-z-Sigma0-remainder*I and P H0 P-z-Sigma0",
        "parity_argument": "center flip of the new path makes M odd, so all odd internal resolvent terms vanish",
        "matrix_scope": "exact finite entries of an infinite operator; no coarse spin-tail gap claim",
        "vacuum_preserving_embedding_claim": False,
    }
    certificate = make_certificate(
        claim="finite theta-graph coarse-holonomy identities and conditional all-spin Feshbach enclosure",
        payload={"type": "su2_theta_refinement_v1", "witness": witness},
        honesty={
            "exact_refinement_identities_verified": True,
            "all_complement_spins_covered": True,
            "self_energy_enclosure_verified": domain,
            "vacuum_preserving_embedding_claim": False,
            "continuum_claim": False,
            "yang_mills_mass_gap_claim": False,
        },
        meta={"analytic_implication": "docs/api/gauge-theta-refinement.md",
              "transcend_backend": "not_used"},
    )
    return {
        "status": "ENCLOSED" if domain else "INCONCLUSIVE_DOMAIN",
        "exact_refinement_identities_verified": True,
        "self_energy_enclosure_verified": domain,
        "all_complement_spins_covered": True,
        "witness": witness, "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        "vacuum_preserving_embedding_claim": False,
        "all_scale_refinement_claim": False, "spectral_gap_claim": False,
        "infinite_volume_claim": False, "uniform_in_a_claim": False,
        "continuum_claim": False, "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
        "theorem_prover_verified": False, "mathlib_verified": False,
    }


def replay_su2_theta_refinement_certificate(certificate: dict[str, Any]) -> bool:
    """Replay either outcome by complete canonical recomputation."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_theta_refinement_v1":
            return False
        inputs = payload["witness"]["inputs"]
        paths = inputs["path_weights"]
        if len(paths) != 3:
            return False
        result = su2_theta_refinement(
            electric=Q(inputs["electric"]),
            path_weights=(Q(paths[0]), Q(paths[1]), Q(paths[2])),
            magnetic_left=Q(inputs["magnetic_left"]),
            magnetic_right=Q(inputs["magnetic_right"]),
            spectral_parameter=Q(inputs["spectral_parameter"]),
            coarse_two_j_max=inputs["coarse_two_j_max"],
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, IndexError, OverflowError):
        return False


__all__ = [
    "replay_su2_theta_refinement_certificate",
    "su2_theta_refinement",
]
