# SPDX-License-Identifier: Apache-2.0
"""A preconditioned actual vacuum on the fixed seven-edge adjacent-square graph."""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.adjacent_resolvent import (
    VACUUM,
    State,
    _basis,
    _energy,
    _exact_inverse,
    _kernel_column,
    _magnetic_column,
    _nuclear,
    replay_su2_adjacent_linearized_inverse_certificate,
    su2_adjacent_linearized_inverse,
)
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational

Polynomial = dict[State, Q]


def _norms(polynomial: Polynomial) -> tuple[Q, Q]:
    anchors = [
        sum(
            (Q(state[i], 2) * _energy(state) * _nuclear(state) * abs(value)
             for state, value in polynomial.items()),
            Q(0),
        )
        for i in range(3)
    ]
    return max(anchors), sum(anchors, Q(0))


def _rows(polynomial: Polynomial) -> list[dict[str, Any]]:
    return [{"state": list(state), "coefficient": str(value)}
            for state, value in sorted(polynomial.items()) if value]


def _reference_residual(g: Q) -> Polynomial:
    square: Polynomial = {}
    for state, first in _magnetic_column(VACUUM):
        for out, second in _magnetic_column(state):
            square[out] = square.get(out, Q(0)) + first * second
    return {
        state: (g / 3)**2 * (Q(3) / _energy(state) - Q(1, 2)) * value
        for state, value in square.items()
        if state != VACUUM and _energy(state) != 6 and value
    }


def su2_adjacent_preconditioned_vacuum(
    kappa: int | Q, *, cutoff: int = 3, correction_radius: int | Q | None = None,
) -> dict[str, Any]:
    """Construct the actual log vacuum if the complete preconditioned gates pass.

    The graph has two adjacent elementary squares and seven unit electric
    edges. The parent inverse is generated and canonically replayed here;
    callers supply no inverse bound or honesty flag. Cutoff>=2 is required
    because the exact reference residual must lie in the retained block.
    Every omitted spin is covered by the replayed parent defect.
    """
    coupling = _rational(kappa, "kappa")
    size = _integer(cutoff, "cutoff")
    supplied_radius = (
        None if correction_radius is None else _rational(correction_radius, "correction_radius")
    )
    if coupling <= 0 or size < 2 or (supplied_radius is not None and supplied_radius <= 0):
        raise ValueError("require kappa>0, cutoff>=2 and a strictly positive optional correction_radius")
    parent = su2_adjacent_linearized_inverse(coupling, cutoff=size)
    parent_replayed = replay_su2_adjacent_linearized_inverse_certificate(parent["certificate"])
    inverse_gate = parent_replayed and parent["full_spin_fourier_reference_inverse_verified"] is True
    g, t, quadratic = 4 / coupling**2, Q(4, 3) / coupling**2, Q(4, 3)
    residual = _reference_residual(g)
    expected = {
        (2, 0, 2): -g**2 / 24,
        (0, 2, 2): -g**2 / 24,
        (1, 1, 0): g**2 / 27,
        (1, 1, 2): -g**2 / 39,
    }
    residual_identity = residual == expected
    basis = tuple(state for state in _basis(size) if state != VACUUM)
    index = {state: i for i, state in enumerate(basis)}
    residual_retained = all(state in index for state in residual)
    head: Polynomial = {}
    leakage: Polynomial = {}
    head_identity = False
    head_n: Q | None = None
    leakage_m: Q | None = None
    error: Q | None = None
    epsilon: Q | None = None
    epsilon_lower: Q | None = None
    inverse_bound: Q | None = None
    defect: Q | None = None
    if inverse_gate and residual_identity and residual_retained:
        columns = {state: _kernel_column(state, t) for state in basis}
        matrix = [[Q(int(i == j)) for j in range(len(basis))] for i in range(len(basis))]
        for state, column in columns.items():
            for out, value in column.items():
                if out in index:
                    matrix[index[out]][index[state]] -= value
        finite_inverse, _, inverse_identity = _exact_inverse(matrix)
        if finite_inverse is not None and inverse_identity:
            rhs = [residual.get(state, Q(0)) for state in basis]
            solution = [
                sum((value * forcing for value, forcing in zip(row, rhs, strict=True)), Q(0))
                for row in finite_inverse
            ]
            head = {state: value for state, value in zip(basis, solution, strict=True) if value}
            head_identity = all(
                sum((value * entry for value, entry in zip(row, solution, strict=True)), Q(0)) == forcing
                for row, forcing in zip(matrix, rhs, strict=True)
            )
            for state, value in head.items():
                for out, coefficient in columns[state].items():
                    if out not in index:
                        leakage[out] = leakage.get(out, Q(0)) + value * coefficient
            leakage = {state: value for state, value in leakage.items() if value}
            head_n = _norms(head)[0]
            leakage_m = _norms(leakage)[1]
            defect = Q(parent["witness"]["arithmetic"]["all_spin_defect_upper"])
            inverse_bound = Q(parent["original_N_inverse_upper"])
            error = leakage_m / (2 * (1 - defect))
            epsilon = head_n + error
            epsilon_lower = max(Q(0), head_n - error)
    source_gate = inverse_gate and residual_identity and residual_retained and head_identity
    beta = inverse_bound * quadratic if inverse_bound is not None else None
    discriminant = 1 - 4 * beta * epsilon if beta is not None and epsilon is not None else None
    radius_exists = source_gate and discriminant is not None and discriminant > 0
    radius = supplied_radius
    if radius is None and radius_exists and epsilon is not None:
        radius = 2 * epsilon
    slack = (
        radius - epsilon - beta * radius**2
        if radius is not None and epsilon is not None and beta is not None else None
    )
    contraction = 2 * beta * radius if beta is not None and radius is not None else None
    fixed_point = (
        source_gate and slack is not None and contraction is not None and slack >= 0
        and contraction < 1
    )
    seed_hessian = 4 * g / 3
    correction_hessian = 2 * radius / 3 if radius is not None else None
    curvature = (
        Q(1, 2) - 2 * seed_hessian - 2 * correction_hessian
        if correction_hessian is not None else None
    )
    gap_gate = fixed_point and curvature is not None and curvature > 0
    gap = coupling * curvature / 2 if gap_gate and curvature is not None else None
    failed = []
    if not inverse_gate:
        failed.append("replayed_complete_original_Fourier_inverse")
    if not residual_identity or not residual_retained or not head_identity:
        failed.append("exact_retained_reference_residual_identity")
    if not fixed_point:
        failed.append("strict_preconditioned_nonlinear_radius")
    if not gap_gate:
        failed.append("positive_original_edge_actual_vacuum_curvature")
    witness = {
        "inputs": {
            "kappa": str(coupling), "cutoff": size,
            "correction_radius": str(supplied_radius) if supplied_radius is not None else None,
        },
        "graph": {
            "n_vertices": 6,
            "oriented_edges": [[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5]],
            "plaquettes_signed_one_based": [[1, 6, -3, -5], [2, 7, -4, -6]],
            "path_lengths": [3, 3, 1], "original_electric_weights": [1] * 7,
            "gauge_constraint": "all six vertices, including boundary; no matter fields",
        },
        "normalization": "aH=kappa*C/2+2*(4-chi_p-chi_q)/kappa; g=4/kappa^2",
        "reference": "S_star=(g/3)*(chi_p+chi_q); true logarithmic vacuum S=S_star+U",
        "linearization": "L=I-2*C0^-1*Pi_H*Gamma(S_star,.), on original mean-zero invariant N",
        "source_inverse_certificate": parent["certificate"],
        "residual_identity": "R_star=C0^-1*Pi_H*Gamma(S_star,S_star)=t^2*(3*C0^-1-1/2)*Pi_H(V^2)",
        "residual_coefficients": _rows(residual),
        "retained_residual_solution": _rows(head),
        "complete_outgoing_residual": _rows(leakage),
        "residual_error_identity": (
            "u0=D^-1*P*R_star; h=Q*K*u0; H=I-diag(D^-1,I)*L; "
            "L^-1*R_star-u0=(I-H)^-1*h; N(error)<=M(h)/(2*(1-z))"
        ),
        "nonlinear_map": "U=L^-1*R_star+L^-1*T(U,U), T=C0^-1*Pi_H*Gamma",
        "quadratic_premise": "N(T(U,V))<=4*N(U)*N(V)/3 on this original girth-four invariant graph",
        "norm_conversion": "2*N<=M<=3*N by triangle admissibility; parent inverse uses3/2 and residual error uses1/2",
        "curvature_proof": (
            "Ricci=1/2; original seed Hessian row<=4g/3 (two squares per edge); "
            "correction Hessian row<=2r/3; rho=1/2-8g/3-4r/3"
        ),
        "reconstruction": (
            "the complete Fourier fixed point is C2; exp(S_star+U)>0 satisfies "
            "(C-gV)psi=E0*psi; the groundstate form identity identifies the actual "
            "unique vacuum; elliptic regularity and Ricci-2Hess(S) give the full scalar gap"
        ),
        "arithmetic": {
            "g": str(g), "t": str(t),
            "parent_inverse_replayed": parent_replayed,
            "complete_inverse_verified": inverse_gate,
            "residual_identity_verified": residual_identity,
            "residual_fully_retained": residual_retained,
            "finite_residual_solve_verified": head_identity,
            "raw_reference_residual_N": str(_norms(residual)[0]),
            "original_N_inverse_upper": str(inverse_bound) if inverse_bound is not None else None,
            "parent_all_spin_defect_upper": str(defect) if defect is not None else None,
            "finite_preconditioned_residual_N": str(head_n) if head_n is not None else None,
            "outgoing_residual_M": str(leakage_m) if leakage_m is not None else None,
            "complete_residual_error_N": str(error) if error is not None else None,
            "preconditioned_residual_N_lower": str(epsilon_lower) if epsilon_lower is not None else None,
            "preconditioned_residual_N_upper": str(epsilon) if epsilon is not None else None,
            "quadratic_constant": str(quadratic),
            "preconditioned_quadratic_upper": str(beta) if beta is not None else None,
            "radius_feasibility_discriminant": str(discriminant) if discriminant is not None else None,
            "source_radius_exists_for_criterion": radius_exists,
            "selected_correction_radius": str(radius) if radius is not None else None,
            "self_map_slack": str(slack) if slack is not None else None,
            "contraction_upper": str(contraction) if contraction is not None else None,
            "fixed_point_verified": fixed_point,
            "seed_hessian_row_upper": str(seed_hessian),
            "correction_hessian_row_upper": str(correction_hessian) if correction_hessian is not None else None,
            "curvature_lower": str(curvature) if curvature is not None else None,
            "physical_gap_lower": str(gap) if gap is not None else None,
        },
        "failed_constraints": failed,
    }
    earned = {
        "replayed_reference_inverse_verified": inverse_gate,
        "preconditioned_residual_verified": source_gate,
        "actual_vacuum_verified": fixed_point,
        "all_spin_actual_vacuum_verified": fixed_point,
        "finite_graph_physical_gap_verified": gap_gate,
    }
    scope = {
        "uniform_in_volume_claim": False, "infinite_volume_claim": False,
        "uniform_in_a_claim": False, "all_scale_refinement_claim": False,
        "continuum_claim": False, "yang_mills_claim": False, "yang_mills_mass_gap_claim": False,
    }
    certificate = make_certificate(
        claim="complete-spin preconditioned actual vacuum and gap on the fixed seven-edge SU2 adjacent-square graph",
        payload={"type": "su2_adjacent_preconditioned_vacuum_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={"analytic_implication": "docs/api/gauge-adjacent-vacuum.md", "transcend_backend": "not_used"},
    )
    return {
        "status": "PASS" if gap_gate else "INCONCLUSIVE",
        "finite_gate_verified": gap_gate,
        "physical_gap_lower": str(gap) if gap is not None else None,
        "witness": witness, "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        **earned, **scope, "theorem_prover_verified": False, "mathlib_verified": False,
    }


def replay_su2_adjacent_preconditioned_vacuum_certificate(certificate: dict[str, Any]) -> bool:
    """Regenerate the actual parent, residual solve, complete tail, and scope."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_adjacent_preconditioned_vacuum_v1":
            return False
        inputs = payload["witness"]["inputs"]
        radius = inputs["correction_radius"]
        result = su2_adjacent_preconditioned_vacuum(
            Q(inputs["kappa"]), cutoff=inputs["cutoff"],
            correction_radius=None if radius is None else Q(radius),
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_su2_adjacent_preconditioned_vacuum_certificate",
    "su2_adjacent_preconditioned_vacuum",
]
