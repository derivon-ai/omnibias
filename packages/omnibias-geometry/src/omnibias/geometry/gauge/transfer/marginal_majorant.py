# SPDX-License-Identifier: Apache-2.0
"""Exact nonnegative matrix majorants under arbitrary subset elimination.

These certificates verify matrix arithmetic. Supplying a matrix does not
establish that it bounds a vacuum Hessian or identify a quantum marginal.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational

Matrix = list[list[Q]]


def _matrix(values: Sequence[Sequence[int | Q]]) -> Matrix:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("matrix must be a nonempty square sequence")
    n = len(values)
    result: Matrix = []
    for row in values:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != n:
            raise ValueError("matrix must be square")
        result.append([_rational(v, "matrix entry") for v in row])
    if any(v < 0 for row in result for v in row):
        raise ValueError("majorant entries must be nonnegative")
    if any(result[i][j] != result[j][i] for i in range(n) for j in range(n)):
        raise ValueError("majorant matrix must be symmetric")
    return result


def _retained(values: Sequence[int], n: int) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("retained must be a sequence of distinct indices")
    result = [_integer(v, "retained index") for v in values]
    if len(set(result)) != len(result) or any(v < 0 or v >= n for v in result):
        raise ValueError("retained indices must be distinct and in range")
    return result


def _distances(values: Sequence[Sequence[int]] | None, n: int) -> list[list[int]] | None:
    if values is None:
        return None
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != n:
        raise ValueError("distances must be a square integer metric")
    result: list[list[int]] = []
    for row in values:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != n:
            raise ValueError("distances must be a square integer metric")
        result.append([_integer(v, "distance") for v in row])
    for i in range(n):
        for j in range(n):
            if result[i][j] != result[j][i] or result[i][j] < (0 if i == j else 1):
                raise ValueError("distances must be a symmetric metric")
            if i == j and result[i][i] != 0:
                raise ValueError("metric diagonal must be zero")
            if any(result[i][j] > result[i][k] + result[k][j] for k in range(n)):
                raise ValueError("distances must obey the triangle inequality")
    return result


def _inverse(matrix: Matrix) -> Matrix:
    n = len(matrix)
    augmented = [list(row) + [Q(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = next((i for i in range(column, n) if augmented[i][column]), None)
        if pivot is None:
            raise ValueError("singular exact matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [v / divisor for v in augmented[column]]
        for i in range(n):
            if i != column:
                factor = augmented[i][column]
                augmented[i] = [x - factor * y for x, y in zip(
                    augmented[i], augmented[column], strict=True)]
    return [row[n:] for row in augmented]


def _transform(matrix: Matrix, retained: list[int], rho: Q) -> tuple[Matrix, Matrix, list[int]]:
    removed = [i for i in range(len(matrix)) if i not in retained]
    block = [[rho * (i == j) - 2 * matrix[i][j] for j in removed] for i in removed]
    kernel = _inverse(block)
    product = [[sum((kernel[a][b] * matrix[removed[b]][j]
                     for b in range(len(removed))), Q(0))
                for j in retained] for a in range(len(removed))]
    output = [[matrix[i][j] + 2 * sum((matrix[i][removed[a]] * product[a][b]
                                     for a in range(len(removed))), Q(0))
               for b, j in enumerate(retained)] for i in retained]
    return output, kernel, removed


def _encode(matrix: Matrix | None) -> list[list[str]] | None:
    return [[str(v) for v in row] for row in matrix] if matrix is not None else None


def marginal_majorant_closure(
    matrix: Sequence[Sequence[int | Q]],
    retained: Sequence[int],
    *,
    ricci_lower: int | Q,
    row_cap: int | Q | None = None,
    distances: Sequence[Sequence[int]] | None = None,
    decay_base: int | Q = 1,
) -> dict[str, Any]:
    """Seal M_CC+2 M_CR (rho I-2 M_RR)^-1 M_RC and its same-cap bound.

    Exact inputs only. The optional integer metric remains the ambient
    metric restricted to retained labels, not a recomputed subgraph metric.
    A deficient supplied row cap or a nonpositive strict Ricci margin is
    an INCONCLUSIVE sufficient gate. Invalid matrix/metric inputs raise.
    """
    original = _matrix(matrix)
    n = len(original)
    keep = _retained(retained, n)
    metric = _distances(distances, n)
    rho, base = _rational(ricci_lower, "ricci_lower"), _rational(decay_base, "decay_base")
    if rho <= 0 or base < 1:
        raise ValueError("ricci_lower must be positive and decay_base at least one")
    if metric is None and base != 1:
        raise ValueError("nontrivial distance weights require an explicit metric")
    weights = [[base**metric[i][j] if metric is not None else Q(1)
                for j in range(n)] for i in range(n)]
    weighted = [[weights[i][j] * original[i][j] for j in range(n)] for i in range(n)]
    rows = [sum(row, Q(0)) for row in weighted]
    cap = max(rows) if row_cap is None else _rational(row_cap, "row_cap")
    if cap < 0:
        raise ValueError("row_cap must be nonnegative")
    contains = max(rows) <= cap
    margin = rho - 2 * cap
    applicable = contains and margin > 0
    output: Matrix | None = None
    kernel: Matrix | None = None
    weighted_output: Matrix | None = None
    comparison: Matrix | None = None
    weighted_kernel: Matrix | None = None
    removed = [i for i in range(n) if i not in keep]
    output_rows: list[Q] | None = None
    savings: list[Q] | None = None
    row_upper: list[Q] | None = None
    checks: dict[str, bool] = {}
    if applicable:
        output, kernel, _ = _transform(original, keep, rho)
        comparison, weighted_kernel, _ = _transform(weighted, keep, rho)
        weighted_output = [[weights[i][j] * output[a][b] for b, j in enumerate(keep)]
                           for a, i in enumerate(keep)]
        output_rows = [sum(row, Q(0)) for row in weighted_output]
        savings = [(1 - 2 * cap / rho) * sum((weighted[i][j] for j in removed), Q(0))
                   for i in keep]
        row_upper = [cap - saving for saving in savings]
        inverse_identity = all(
            sum(((rho * (i == k) - 2 * original[i][k]) * kernel[a][b]
                 for a, k in enumerate(removed)), Q(0)) == (i == j)
            for i in removed for b, j in enumerate(removed)
        )
        weighted_inverse_identity = all(
            sum(((rho * (i == k) - 2 * weighted[i][k]) * weighted_kernel[a][b]
                 for a, k in enumerate(removed)), Q(0)) == (i == j)
            for i in removed for b, j in enumerate(removed)
        )
        checks = {
            "kernel_identity_verified": inverse_identity,
            "weighted_kernel_identity_verified": weighted_inverse_identity,
            "kernels_nonnegative_verified": all(v >= 0 for row in kernel + weighted_kernel for v in row),
            "output_nonnegative_symmetric_verified": all(
                output[i][j] >= 0 and output[i][j] == output[j][i]
                for i in range(len(keep)) for j in range(len(keep))),
            "weighted_path_domination_verified": all(
                weighted_output[i][j] <= comparison[i][j]
                for i in range(len(keep)) for j in range(len(keep))),
            "strict_row_savings_verified": all(
                sum(comparison[i], Q(0)) <= row_upper[i] for i in range(len(keep))),
            "same_weighted_row_cap_verified": all(v <= cap for v in output_rows),
        }
    passed = applicable and all(checks.values())
    failures = []
    if not contains:
        failures.append("supplied_row_cap_below_actual_weighted_rows")
    if margin <= 0:
        failures.append("strict_ricci_margin")
    failures.extend(key for key, value in checks.items() if not value)
    witness = {
        "inputs": {"matrix": _encode(original), "retained": keep,
                   "ricci_lower": str(rho), "row_cap": str(cap) if row_cap is not None else None,
                   "distances": metric, "decay_base": str(base)},
        "removed": removed,
        "input_weighted_row_sums": [str(v) for v in rows],
        "effective_row_cap": str(cap), "row_cap_contains_input_verified": contains,
        "strict_ricci_margin": str(margin),
        "inverse_kernel": _encode(kernel), "weighted_inverse_kernel": _encode(weighted_kernel),
        "output_matrix": _encode(output), "weighted_output_matrix": _encode(weighted_output),
        "weighted_schur_comparison_matrix": _encode(comparison),
        "output_weighted_row_sums": [str(v) for v in output_rows] if output_rows is not None else None,
        "row_savings_lower": [str(v) for v in savings] if savings is not None else None,
        "output_weighted_row_upper": [str(v) for v in row_upper] if row_upper is not None else None,
        "retained_distances": [[metric[i][j] for j in keep] for i in keep] if metric is not None else None,
        "transform": "M_CC+2*M_CR*(rho*I-2*M_RR)^-1*M_RC",
        "schur_relation": "rho*I-2*M_out is the Schur complement of rho*I-2*M",
        "metric_scope": "original ambient integer metric restricted to retained labels; no induced-subgraph distance substitution",
        "analytic_premises": ["a supplied majorant must separately be shown to bound the actual Hessian blocks",
                              "the conditional matrix covariance inequality must separately apply to the actual measure",
                              "a changed kinetic metric or physical scale requires its own bounds"],
        "scope": "finite exact symmetric nonnegative matrix arithmetic; no vacuum, Hamiltonian or physical refinement is inferred from matrix entries",
        "checks": checks, "failed_constraints": failures,
    }
    scope = {"actual_vacuum_hessian_verified": False, "actual_marginal_verified": False,
             "all_scale_refinement_claim": False, "infinite_volume_claim": False,
             "uniform_in_a_claim": False, "continuum_claim": False,
             "yang_mills_mass_gap_claim": False}
    certificate = make_certificate(
        claim="exact finite marginal majorant transform preserves its weighted row cap under a positive strict Ricci margin",
        payload={"type": "marginal_majorant_closure_v1", "witness": witness},
        honesty={"finite_matrix_majorant_closure_verified": passed, **scope},
        meta={"analytic_implication": "docs/api/gauge-marginal-majorant.md",
              "transcend_backend": "not_used"},
    )
    return {"status": "PASS" if passed else "INCONCLUSIVE", "finite_gate_verified": passed,
            "finite_matrix_majorant_closure_verified": passed,
            "output_matrix": _encode(output), "witness": witness, "certificate": certificate,
            "digest_verified": verify_certificate_digest(certificate), **scope,
            "theorem_prover_verified": False, "mathlib_verified": False}


def replay_marginal_majorant_closure_certificate(certificate: dict[str, Any]) -> bool:
    """Recompute the matrix, metric, inverse, output and all scope fields."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "marginal_majorant_closure_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = marginal_majorant_closure(
            [[Q(v) for v in row] for row in inputs["matrix"]], inputs["retained"],
            ricci_lower=Q(inputs["ricci_lower"]),
            row_cap=Q(inputs["row_cap"]) if inputs["row_cap"] is not None else None,
            distances=inputs["distances"], decay_base=Q(inputs["decay_base"]),
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


def conditional_poincare_schur(
    gaps: Sequence[int | Q],
    mixed_hessian: Sequence[Sequence[int | Q]],
    retained: Sequence[int],
    *,
    distances: Sequence[Sequence[int]] | None = None,
    decay_base: int | Q = 1,
) -> dict[str, Any]:
    """Exact weighted Schur arithmetic with unequal conditional-gap inputs.

    The supplied positive gaps and off-diagonal Hessian bounds are
    analytic premises, not earned facts about a measure. The reference
    rho=max(gaps) is an algebraic normalization, not a Ricci assertion.
    Failure of strict weighted diagonal dominance is INCONCLUSIVE.
    """
    if not isinstance(gaps, Sequence) or isinstance(gaps, (str, bytes)) or not gaps:
        raise ValueError("gaps must be a nonempty sequence of positive exact rationals")
    gamma = [_rational(value, "conditional gap") for value in gaps]
    if any(value <= 0 for value in gamma):
        raise ValueError("conditional gaps must be strictly positive")
    mixed = _matrix(mixed_hessian)
    n = len(gamma)
    if len(mixed) != n or any(mixed[i][i] != 0 for i in range(n)):
        raise ValueError("mixed_hessian must match gaps and have zero diagonal")
    rho = max(gamma)
    majorant = [[(rho - gamma[i]) / 2 if i == j else mixed[i][j]
                 for j in range(n)] for i in range(n)]
    source = marginal_majorant_closure(
        majorant, retained, ricci_lower=rho, distances=distances, decay_base=decay_base,
    )
    source_witness = source["witness"]
    source_inputs = source_witness["inputs"]
    keep: list[int] = source_inputs["retained"]
    metric: list[list[int]] | None = source_inputs["distances"]
    base = Q(source_inputs["decay_base"])
    weights = [[base**metric[i][j] if metric is not None else Q(1)
                for j in range(n)] for i in range(n)]
    margins = [gamma[i] - 2 * sum((weights[i][j] * mixed[i][j] for j in range(n)), Q(0))
               for i in range(n)]
    delta = min(margins)
    checks = {
        "source_strict_margin_identity_verified": Q(source_witness["strict_ricci_margin"]) == delta,
        "weighted_diagonal_dominance_verified": delta > 0,
        "finite_matrix_source_verified": bool(source["finite_gate_verified"]),
    }
    output_gaps: list[Q] | None = None
    output_mixed: Matrix | None = None
    output_h: Matrix | None = None
    output_margins: list[Q] | None = None
    if source["finite_gate_verified"]:
        output = [[Q(v) for v in row] for row in source["output_matrix"]]
        output_gaps = [rho - 2 * output[i][i] for i in range(len(keep))]
        output_mixed = [[Q(0) if i == j else output[i][j]
                         for j in range(len(keep))] for i in range(len(keep))]
        output_h = [[output_gaps[i] if i == j else -2 * output_mixed[i][j]
                     for j in range(len(keep))] for i in range(len(keep))]
        output_margins = [output_gaps[i] - 2 * sum(
            (weights[keep[i]][keep[j]] * output_mixed[i][j] for j in range(len(keep))), Q(0))
            for i in range(len(keep))]
        checks.update({
            "output_gaps_positive_verified": all(v > 0 for v in output_gaps),
            "output_mixed_nonnegative_symmetric_verified": all(
                output_mixed[i][j] >= 0 and output_mixed[i][j] == output_mixed[j][i]
                for i in range(len(keep)) for j in range(len(keep))),
            "retained_weighted_dominance_preserved_verified": all(v >= delta for v in output_margins),
        })
    passed = all(checks.values())
    scope = {
        "actual_conditional_poincare_verified": False,
        "actual_mixed_hessian_verified": False, "actual_vacuum_hessian_verified": False,
        "actual_marginal_verified": False, "physical_gap_verified": False,
        "all_scale_refinement_claim": False, "infinite_volume_claim": False,
        "uniform_in_a_claim": False, "continuum_claim": False,
        "yang_mills_mass_gap_claim": False,
    }
    witness = {
        "inputs": {"gaps": [str(v) for v in gamma], "mixed_hessian": _encode(mixed),
                   "retained": keep, "distances": metric, "decay_base": str(base)},
        "reference_rho": str(rho),
        "reference_scope": "algebraic rho=max(gaps), not a geometric Ricci lower bound",
        "input_comparison_matrix": _encode([
            [gamma[i] if i == j else -2 * mixed[i][j] for j in range(n)] for i in range(n)]),
        "encoded_majorant": _encode(majorant),
        "input_weighted_dominance_margins": [str(v) for v in margins],
        "uniform_dominance_lower": str(delta),
        "source_certificate": source["certificate"],
        "source_use": "exact finite matrix inverse, Schur transform and weighted row identity only",
        "output_gaps": [str(v) for v in output_gaps] if output_gaps is not None else None,
        "output_mixed_hessian": _encode(output_mixed), "output_comparison_matrix": _encode(output_h),
        "output_weighted_dominance_margins": [str(v) for v in output_margins]
        if output_margins is not None else None,
        "retained_distances": source_witness["retained_distances"],
        "transform": "H_CC-H_CR*H_RR^-1*H_RC, H_ii=gamma_i and H_ij=-2*c_ij",
        "normalization_invariance": "replacing reference rho on a later step does not change H or its Schur complement",
        "analytic_premises": [
            "each supplied gamma_i must separately bound the actual conditional Poincare gap",
            "each supplied c_ij must separately bound the actual mixed log-vacuum Hessian block",
            "the conditional one-form covariance theorem must apply to the actual measure and domains",
            "physical kinetic and scale identifications require their own proof",
        ],
        "scope": "finite exact conditional-gap comparison-matrix arithmetic; supplied analytic premises are unverified",
        "checks": checks, "failed_constraints": [key for key, value in checks.items() if not value],
    }
    certificate = make_certificate(
        claim="strict weighted diagonal dominance and its positive margin survive this exact finite conditional-gap Schur transform",
        payload={"type": "conditional_poincare_schur_v1", "witness": witness},
        honesty={"finite_conditional_poincare_schur_verified": passed, **scope},
        meta={"analytic_implication": "docs/api/gauge-marginal-majorant.md",
              "transcend_backend": "not_used"},
    )
    return {
        "status": "PASS" if passed else "INCONCLUSIVE", "finite_gate_verified": passed,
        "finite_conditional_poincare_schur_verified": passed,
        "output_gaps": witness["output_gaps"], "output_mixed_hessian": _encode(output_mixed),
        "witness": witness, "certificate": certificate, **scope,
        "digest_verified": verify_certificate_digest(certificate),
        "theorem_prover_verified": False, "mathlib_verified": False,
    }


def replay_conditional_poincare_schur_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild all inputs, the nested matrix source, output and scope."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "conditional_poincare_schur_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = conditional_poincare_schur(
            [Q(v) for v in inputs["gaps"]],
            [[Q(v) for v in row] for row in inputs["mixed_hessian"]],
            inputs["retained"], distances=inputs["distances"], decay_base=Q(inputs["decay_base"]),
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "conditional_poincare_schur",
    "marginal_majorant_closure",
    "replay_conditional_poincare_schur_certificate",
    "replay_marginal_majorant_closure_certificate",
]
