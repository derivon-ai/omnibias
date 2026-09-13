# SPDX-License-Identifier: Apache-2.0
"""Exact retained inverses with complete tails for an adjacent SU2 reference.

The seven-edge theta electric operator is retained, including its shared
spin. This is a reference-linearization inverse, never a target-vacuum gap.
The scalar coefficient norm is related explicitly to the original three
anchored Fourier sums; all omitted spins are bounded analytically.
"""

from __future__ import annotations

from fractions import Fraction as Q
from functools import lru_cache
from itertools import product
from math import lcm
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.sixj import _triangle, racah_sixj_squared
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational

State = tuple[int, int, int]
Matrix = list[list[Q]]
VACUUM: State = (0, 0, 0)


def _basis(cutoff: int) -> tuple[State, ...]:
    return tuple((s[0], s[1], s[2]) for s in product(range(cutoff + 1), repeat=3) if _triangle(*s))


def _energy(state: State) -> Q:
    a, b, s = state
    return Q(3 * a * (a + 2) + 3 * b * (b + 2) + s * (s + 2), 4)


def _nuclear(state: State) -> int:
    return (state[0] + 1) ** 2 * (state[1] + 1) ** 2


def _weight(state: State) -> Q:
    return Q(sum(state), 2) * _energy(state) * _nuclear(state)


@lru_cache(maxsize=32768)
def _magnetic_column(state: State) -> tuple[tuple[State, Q], ...]:
    """All coefficients of (chi_p+chi_q)b_state, including its Haar constant."""
    entries: dict[State, Q] = {}
    for active in (0, 1):
        spectator = 1 - active
        for step, shared_step in product((-1, 1), repeat=2):
            labels = list(state)
            labels[active] += step
            labels[2] += shared_step
            outside: State = (labels[0], labels[1], labels[2])
            if not _triangle(*outside):
                continue
            entry = (
                (outside[active] + 1)
                * (outside[2] + 1)
                * racah_sixj_squared(
                    outside[active], outside[2], outside[spectator], state[2], state[active], 1
                )
            )
            if entry:
                entries[outside] = entries.get(outside, Q(0)) + entry
    return tuple(sorted(entries.items()))


def _kernel_column(state: State, t: Q) -> dict[State, Q]:
    energy = _energy(state)
    return {
        out: t * (3 + energy - _energy(out)) * value / _energy(out)
        for out, value in _magnetic_column(state)
        if out != VACUUM and 3 + energy != _energy(out)
    }


def _exact_inverse(matrix: Matrix) -> tuple[Matrix | None, list[int], bool]:
    """Fraction-free Gauss--Jordan with exact division and sparse residual check."""
    size = len(matrix)
    scales = [lcm(*(x.denominator for x in row)) for row in matrix]
    aug = [
        [int(scale * x) for x in row] + [scale * int(i == j) for j in range(size)]
        for i, (row, scale) in enumerate(zip(matrix, scales, strict=True))
    ]
    previous = 1
    pivots: list[int] = []
    for k in range(size):
        pivot_row = next((i for i in range(k, size) if aug[i][k]), None)
        if pivot_row is None:
            return None, pivots, False
        aug[k], aug[pivot_row] = aug[pivot_row], aug[k]
        pivot = aug[k][k]
        pivots.append(pivot)
        for i in range(size):
            if i == k:
                continue
            factor = aug[i][k]
            for j in range(2 * size):
                if j == k:
                    continue
                numerator = pivot * aug[i][j] - factor * aug[k][j]
                quotient, remainder = divmod(numerator, previous)
                if remainder:
                    return None, pivots, False
                aug[i][j] = quotient
            aug[i][k] = 0
        previous = pivot
    inverse = [[Q(x, aug[i][i]) for x in row[size:]] for i, row in enumerate(aug)]
    sparse_rows = [[(k, value) for k, value in enumerate(row) if value] for row in matrix]
    verified = all(
        sum((value * inverse[k][j] for k, value in row), Q(0)) == int(i == j)
        for i, row in enumerate(sparse_rows)
        for j in range(size)
    )
    return inverse, pivots, verified


def _tail(t: Q, label: int) -> Q:
    return 256 * t * Q((label + 1) * (label + 2), label**2 * (5 * label + 16))


def su2_adjacent_linearized_inverse(
    kappa: int | Q,
    *,
    cutoff: int = 3,
) -> dict[str, Any]:
    """Certify a full-spin reference inverse using a complete retained theta block.

    Retain all admissible doubled spins <=cutoff except the Haar constant.
    The finite block is inverted over Q. Every boundary column is recomputed;
    a proved monotone bound covers all further columns. Failed finite inverses
    or defects >=1 return INCONCLUSIVE. The original anchored norm bound
    includes an explicit factor-3/2 conversion from the scalar l1 norm.
    """
    coupling = _rational(kappa, "kappa")
    size = _integer(cutoff, "cutoff")
    if coupling <= 0 or size <= 0:
        raise ValueError("kappa and cutoff must be strictly positive")
    g, t = 4 / coupling**2, Q(4, 3) / coupling**2
    basis = tuple(s for s in _basis(size) if s != VACUUM)
    index = {state: i for i, state in enumerate(basis)}
    matrix = [[Q(i == j) for j in range(len(basis))] for i in range(len(basis))]
    for j, state in enumerate(basis):
        for out, value in _kernel_column(state, t).items():
            if out in index:
                matrix[index[out]][j] -= value
    inverse, pivots, finite_verified = _exact_inverse(matrix)
    boundary = tuple(s for s in _basis(size + 1) if max(s) == size + 1)
    examined = (*basis, *boundary)
    multiplication_verified = all(
        sum((value for _, value in _magnetic_column(state)), Q(0)) == 4
        and all(value >= 0 for _, value in _magnetic_column(state))
        for state in examined
    )
    columns: list[Q] = []
    retained: list[Q] = []
    near: list[Q] = []
    if inverse is not None:
        columns = [
            sum((_weight(state) * abs(inverse[i][j]) for i, state in enumerate(basis)), Q(0))
            / _weight(source)
            for j, source in enumerate(basis)
        ]
        retained = [
            sum(
                (
                    _weight(out) * abs(value)
                    for out, value in _kernel_column(state, t).items()
                    if out not in index
                ),
                Q(0),
            )
            / _weight(state)
            for state in basis
        ]
        for state in boundary:
            column = _kernel_column(state, t)
            incoming = [(index[out], value) for out, value in column.items() if out in index]
            inside = [
                sum((inverse[i][j] * value for j, value in incoming), Q(0))
                for i in range(len(basis))
            ]
            value = sum((_weight(s) * abs(v) for s, v in zip(basis, inside, strict=True)), Q(0))
            value += sum(
                (_weight(out) * abs(v) for out, v in column.items() if out not in index), Q(0)
            )
            near.append(value / _weight(state))
    far = _tail(t, size + 2)
    defect = max(max(retained), max(near), far) if inverse is not None else None
    preconditioner = max(Q(1), max(columns)) if inverse is not None else None
    passed = finite_verified and multiplication_verified and defect is not None and defect < 1
    scalar_bound = (
        preconditioner / (1 - defect)
        if passed and preconditioner is not None and defect is not None
        else None
    )
    anchored_bound = Q(3, 2) * scalar_bound if scalar_bound is not None else None
    earned = {
        "finite_reference_block_inverse_verified": finite_verified,
        "finite_magnetic_identity_verified": multiplication_verified,
        "full_spin_scalar_reference_inverse_verified": passed,
        "full_spin_fourier_reference_inverse_verified": passed,
        "reference_inverse_verified": passed,
    }
    scope = {
        "actual_vacuum_verified": False,
        "actual_vacuum_claim": False,
        "physical_mass_gap_claim": False,
        "uniform_in_volume_claim": False,
        "infinite_volume_claim": False,
        "uniform_in_a_claim": False,
        "all_scale_refinement_claim": False,
        "spatial_locality_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
    }
    witness = {
        "inputs": {"kappa": str(coupling), "cutoff": size},
        "model": "neutral seven-edge SU2 adjacent plaquettes, all theta spins",
        "normalization": (
            "aH=kappa*C/2+2*(4-chi_p-chi_q)/kappa; g=4/kappa^2; S*=(g/3)*(chi_p+chi_q)"
        ),
        "operator": "L=I-K; K=2*C0^-1*Pi_H*Gamma(S*,.); zero Haar mean",
        "basis_definition": "b_(a,b,s)=Tr(P_s(D_a tensor D_b))/(s+1), doubled labels; b(I)=1",
        "electric_energy": "E=(3*a*(a+2)+3*b*(b+2)+s*(s+2))/4, seven unit-weight edges",
        "original_coefficient_nuclear_norm": (
            "(a+1)^2*(b+1)^2 in canonical rightward/upward edge orientation"
        ),
        "scalar_norm_M": "sum_(a,b,s)!=0 (a+b+s)*E*(a+1)^2*(b+1)^2*abs(h_(a,b,s))/2",
        "original_norm_N": "max_(i=a,b,s) sum_(a,b,s)!=0 i*E*(a+1)^2*(b+1)^2*abs(h_(a,b,s))/2",
        "norm_conversion": (
            "2*N <= M <= 3*N by the theta triangle inequalities; "
            "unweighted diameter b=1 only"
        ),
        "magnetic_coefficient": (
            "m_out,in=(out_active+1)*(out_s+1)*"
            "sixj(out_active,out_s,spectator;in_s,in_active,1)^2; "
            "complete +-1 active/shared neighbors"
        ),
        "linearized_coefficient": (
            "K_out,in=t*(3+E_in-E_out)*m_out,in/E_out; omit constant output"
        ),
        "far_tail_formula": "256*t*(J+1)*(J+2)/(J^2*(5*J+16)), decreasing for J>=1",
        "tail_quantifier": "all input max(a,b,s)>=cutoff+2; all outputs included",
        "basis": [list(s) for s in basis],
        "boundary_basis": [list(s) for s in boundary],
        "arithmetic": {
            "g": str(g),
            "t": str(t),
            "retained_dimension": len(basis),
            "boundary_dimension": len(boundary),
            "retained_inverse_exact_pivots": [str(p) for p in pivots],
            "finite_inverse_residual_verified": finite_verified,
            "magnetic_nonnegative_mass_four_verified": multiplication_verified,
            "retained_inverse_weighted_column_norms": [str(c) for c in columns],
            "preconditioner_norm": str(preconditioner) if preconditioner is not None else None,
            "retained_defect_columns": [str(c) for c in retained],
            "boundary_defect_columns": [str(c) for c in near],
            "retained_defect_upper": str(max(retained)) if retained else None,
            "boundary_defect_upper": str(max(near)) if near else None,
            "far_tail_first_label": size + 2,
            "far_tail_upper": str(far),
            "all_spin_defect_upper": str(defect) if defect is not None else None,
            "strict_defect_slack": str(1 - defect) if defect is not None else None,
            "scalar_M_inverse_upper": str(scalar_bound) if scalar_bound is not None else None,
            "norm_conversion_factor": "3/2",
            "original_N_inverse_upper": str(anchored_bound) if anchored_bound is not None else None,
            "original_N_single_column_norm_lower": str(Q(13, 2) * t),
            "scalar_M_single_column_norm_lower": str(Q(15, 2) * t),
            "old_strip_linear_majorant": str(Q(56, 3) * g),
        },
        "failed_constraints": ([] if passed else ["strict_complete_spin_preconditioned_defect"]),
    }
    certificate = make_certificate(
        claim=(
            "exact retained-block and complete-spin-tail gates for the original anchored "
            "Fourier inverse of an explicit adjacent-plaquette reference linearization"
        ),
        payload={"type": "su2_adjacent_linearized_inverse_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={
            "analytic_implication": "docs/api/gauge-adjacent-resolvent.md",
            "transcend_backend": "not_used",
        },
    )
    return {
        "status": "PASS" if passed else "INCONCLUSIVE",
        "finite_gate_verified": passed,
        "scalar_M_inverse_upper": str(scalar_bound) if scalar_bound is not None else None,
        "original_N_inverse_upper": str(anchored_bound) if anchored_bound is not None else None,
        "witness": witness,
        "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        **earned,
        **scope,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


def replay_su2_adjacent_linearized_inverse_certificate(certificate: dict[str, Any]) -> bool:
    """Recompute canonical passing or inconclusive finite-block/all-tail arithmetic."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_adjacent_linearized_inverse_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = su2_adjacent_linearized_inverse(Q(inputs["kappa"]), cutoff=inputs["cutoff"])
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = ["replay_su2_adjacent_linearized_inverse_certificate", "su2_adjacent_linearized_inverse"]
