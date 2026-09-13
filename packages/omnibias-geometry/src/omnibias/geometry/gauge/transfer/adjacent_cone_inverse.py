# SPDX-License-Identifier: Apache-2.0
"""Original theta norm inverse bounds from exact directional dual weights."""
from __future__ import annotations

from fractions import Fraction as Q
from itertools import combinations
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.adjacent_resolvent import (
    VACUUM,
    State,
    _basis,
    _energy,
    _exact_inverse,
    _kernel_column,
    _nuclear,
    _tail,
    replay_su2_adjacent_linearized_inverse_certificate,
    su2_adjacent_linearized_inverse,
)
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational

Weights = tuple[Q, Q, Q]
Constraint = tuple[Weights, Q]


def _anchors(state: State) -> Weights:
    common = _energy(state) * _nuclear(state) / 2
    return (common * state[0], common * state[1], common * state[2])


def _output_anchors(column: dict[State, Q]) -> Weights:
    values = [
        sum((_anchors(state)[i] * abs(value) for state, value in column.items()), Q(0))
        for i in range(3)
    ]
    return (values[0], values[1], values[2])


def _dot(a: Weights, b: Weights) -> Q:
    return sum((x * y for x, y in zip(a, b, strict=True)), Q(0))


def _dual_in_family(constraints: list[Constraint], first: Weights, second: Weights) -> dict[str, Any]:
    """Exact two-parameter search; only feasibility, not global optimality, is earned."""
    rows = [(_dot(a, first), _dot(a, second), bound) for a, bound in constraints]
    assert all(a + b > 0 for a, b, _ in rows)
    uniform = max([Q(0), *(c / (a + b) for a, b, c in rows)])
    best = (uniform, uniform)
    cost_x, cost_y = sum(first, Q(0)), sum(second, Q(0))
    best_cost = uniform * (cost_x + cost_y)
    boundaries = [*rows, (Q(1), Q(0), Q(0)), (Q(0), Q(1), Q(0))]
    for (a, b, c), (d, e, f) in combinations(boundaries, 2):
        det = a * e - b * d
        if not det:
            continue
        x, y = (c * e - b * f) / det, (a * f - c * d) / det
        cost = cost_x * x + cost_y * y
        if x < 0 or y < 0 or cost >= best_cost:
            continue
        if all(left * x + right * y >= bound for left, right, bound in rows):
            best, best_cost = (x, y), cost
    values = [
        best[0] * a + best[1] * b for a, b in zip(first, second, strict=True)
    ]
    weights: Weights = (values[0], values[1], values[2])
    slacks = [_dot(a, weights) - bound for a, bound in constraints]
    assert all(v >= 0 for v in weights) and all(v >= 0 for v in slacks)
    return {"weights": [str(v) for v in weights], "sum": str(sum(weights, Q(0))),
            "constraint_slacks": [str(v) for v in slacks],
            "constraint_rows": [{"weights": [str(v) for v in a], "lower": str(b)} for a, b in constraints],
            "dual_feasibility_verified": True, "global_dual_optimality_claim": False}


def su2_adjacent_cone_inverse(kappa: int | Q, *, cutoff: int = 3) -> dict[str, Any]:
    """Bound L^-1 by R plus its complete defect, keeping each original anchor."""
    coupling, size = _rational(kappa, "kappa"), _integer(cutoff, "cutoff")
    if coupling <= 0 or size < 1:
        raise ValueError("require kappa>0 and cutoff>=1")
    parent = su2_adjacent_linearized_inverse(coupling, cutoff=size)
    parent_replayed = replay_su2_adjacent_linearized_inverse_certificate(parent["certificate"])
    passed = parent_replayed and parent["full_spin_fourier_reference_inverse_verified"] is True
    pre_duals: list[dict[str, Any]] = []
    mixed_dual: dict[str, Any] | None = None
    pre_norm: Q | None = None
    mixed_norm: Q | None = None
    bound: Q | None = None
    if passed:
        states = tuple(s for s in _basis(size) if s != VACUUM)
        boundary = tuple(s for s in _basis(size + 1) if max(s) == size + 1)
        index = {state: i for i, state in enumerate(states)}
        t = Q(4, 3) / coupling**2
        columns = {state: _kernel_column(state, t) for state in (*states, *boundary)}
        matrix = [[Q(i == j) for j in range(len(states))] for i in range(len(states))]
        for j, state in enumerate(states):
            for out, value in columns[state].items():
                if out in index:
                    matrix[index[out]][j] -= value
        inverse, _, checked = _exact_inverse(matrix)
        if inverse is None or not checked:
            raise ArithmeticError("replayed retained inverse did not reproduce")
        r_columns = [{s: inverse[i][j] for i, s in enumerate(states) if inverse[i][j]}
                     for j in range(len(states))]
        output_norms = [_output_anchors(col) for col in r_columns]
        pairs = tuple(combinations(range(3), 2))
        directions: list[tuple[Weights, Weights]] = [
            ((Q(1), Q(0), Q(0)), (Q(0), Q(0), Q(1))),
            ((Q(0), Q(1), Q(0)), (Q(0), Q(0), Q(1))),
            ((Q(1), Q(1), Q(0)), (Q(0), Q(0), Q(1))),
        ]
        for anchor in range(3):
            constraints = [(_anchors(s), norms[anchor]) for s, norms in zip(states, output_norms, strict=True)]
            # Identity on every omitted spin: the three extreme triangle rays.
            for pair in pairs:
                ray: Weights = (Q(0 in pair), Q(1 in pair), Q(2 in pair))
                constraints.append((ray, Q(anchor in pair)))
            pre_duals.append(_dual_in_family(constraints, *directions[anchor]))
        mixed: list[Constraint] = []
        for state, col in zip(states, r_columns, strict=True):
            outgoing: dict[State, Q] = {}
            for middle, value in col.items():
                for out, coefficient in columns[middle].items():
                    if out not in index:
                        outgoing[out] = outgoing.get(out, Q(0)) + value * coefficient
            mixed.append((_anchors(state), sum(_output_anchors(outgoing), Q(0))))
        for state in boundary:
            raw = columns[state]
            retained = [
                sum((inverse[i][index[out]] * v for out, v in raw.items() if out in index), Q(0))
                for i in range(len(states))
            ]
            total = {out: value for out, value in raw.items() if out not in index}
            total.update({out: v for out, v in zip(states, retained, strict=True) if v})
            mixed.append((_anchors(state), sum(_output_anchors(total), Q(0))))
        far = _tail(t, size + 2)
        for pair in pairs:
            ray = (Q(0 in pair), Q(1 in pair), Q(2 in pair))
            mixed.append((ray, 2 * far))
        mixed_dual = _dual_in_family(mixed, *directions[2])
        pre_norm = max(Q(row["sum"]) for row in pre_duals)
        mixed_norm = Q(mixed_dual["sum"])
        z = Q(parent["witness"]["arithmetic"]["all_spin_defect_upper"])
        bound = pre_norm + mixed_norm / (2 * (1 - z))
    witness = {
        "inputs": {"kappa": str(coupling), "cutoff": size},
        "source_inverse_certificate": parent["certificate"],
        "model": "fixed seven-edge adjacent-square SU2 reference, every theta spin",
        "norm": "original N=max_i sum_j w_ij*abs(c_j); 2N<=M=sum_i N_i<=3N",
        "dual_lemma": "lambda>=0 and sum_i lambda_i*w_ij>=c_j for all columns imply sum_j c_j*abs(x_j)<=sum_i lambda_i*N(x)",
        "cone_tail_lemma": "nonnegative triangle spins are generated by rays110,101,011; all tail dual constraints reduce to three pair sums",
        "preconditioner": "R=diag((P*L*P)^-1,I); Z=I-R*L",
        "inverse_identity": "L^-1=R+(I-Z)^-1*Z*R",
        "mixed_operator": "Z*R: original N input to scalar M output; retained columns are Q*K*R_P, boundary columns are Z, all farther columns are K",
        "complete_inverse_bound": "||L^-1||_N<=||R||_N+||Z*R||_(N to M)/(2*(1-z))",
        "preconditioner_duals": pre_duals,
        "mixed_defect_dual": mixed_dual,
        "arithmetic": {
            "parent_replayed": parent_replayed,
            "preconditioner_N_upper": str(pre_norm) if pre_norm is not None else None,
            "mixed_ZR_N_to_M_upper": str(mixed_norm) if mixed_norm is not None else None,
            "original_N_inverse_upper": str(bound) if bound is not None else None,
        },
        "failed_constraints": [] if passed else ["complete_parent_inverse"],
    }
    earned = {"full_spin_fourier_reference_inverse_verified": passed,
              "directional_dual_weights_verified": passed}
    scope = {"actual_vacuum_verified": False, "physical_mass_gap_claim": False,
             "uniform_in_volume_claim": False, "infinite_volume_claim": False,
             "uniform_in_a_claim": False, "continuum_claim": False,
             "yang_mills_claim": False, "yang_mills_mass_gap_claim": False}
    certificate = make_certificate(
        claim="complete original theta inverse via rational directional dual weights and all-spin tails",
        payload={"type": "su2_adjacent_cone_inverse_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={"analytic_implication": "docs/api/gauge-adjacent-cone-inverse.md", "transcend_backend": "not_used"},
    )
    return {"status": "PASS" if passed else "INCONCLUSIVE",
            "original_N_inverse_upper": str(bound) if bound is not None else None,
            "witness": witness, "certificate": certificate,
            "digest_verified": verify_certificate_digest(certificate),
            **earned, **scope, "theorem_prover_verified": False, "mathlib_verified": False}


def replay_su2_adjacent_cone_inverse_certificate(certificate: dict[str, Any]) -> bool:
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_adjacent_cone_inverse_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = su2_adjacent_cone_inverse(Q(inputs["kappa"]), cutoff=inputs["cutoff"])
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError, ArithmeticError):
        return False


__all__ = [
    "replay_su2_adjacent_cone_inverse_certificate",
    "su2_adjacent_cone_inverse",
]
