# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact finite weighted matrix bounds and a conditional quadratic ball gate.

Only integers and Fractions are accepted. No numerical or derivative backend is
imported. A matrix result concerns the supplied finite matrix, and the quadratic
gate does not establish its external Banach-space/operator hypotheses. Neither
result verifies a mixed-Hessian PDE estimate, a continuum theorem, or a Lean tier.

For a rectangular matrix A, the weighted transform is D_row A D_col**(-1).
The maximum absolute row and column sums R,C give ||.||_2**2 <= R*C.

For symmetric K and an integer metric d, Gershgorin gives K >= lambda*I,
where lambda=min_i(K_ii-sum_(j!=i)|K_ij|). For any 1-Lipschitz height h
in metric d, conjugation by D=diag(w**h_i) differs from K by norm at most
eta=max_i sum_(j!=i)|K_ij|*(w**d_ij-1). The symmetric magnitude majorant
bounds both row and column sums, so its l2 bound is eta. The weighted
semigroup decay is lambda-eta. Acting on a matrix on both sides therefore
gives decay 2*(lambda-eta), while conjugating only one side gives 2*lambda-eta.
These finite matrix statements do not identify a PDE Hessian with that model.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from fractions import Fraction
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest

Rational = int | Fraction
_SCHUR = "finite_weighted_schur_exact_v1"
_GENERATOR = "finite_weighted_symmetric_generator_exact_v1"
_BALL = "conditional_quadratic_ball_exact_v1"
_SCOPE = {
    "verification_kind": "EXACT_RATIONAL_FINITE_ARITHMETIC",
    "mixed_hessian_pde_locality_verified": False,
    "nonlinear_pde_closure_verified": False,
    "infinite_volume_claim": False,
    "continuum_claim": False,
    "theorem_prover_verified": False,
    "mathlib_verified": False,
}


def _rational(value: Rational, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"{name} must be an integer or Fraction")
    return Fraction(value)


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a finite sequence")
    return value


def _matrix(matrix: Sequence[Sequence[Rational]]) -> tuple[tuple[Fraction, ...], ...]:
    rows = _sequence(matrix, "matrix")
    if not rows:
        raise ValueError("matrix must have at least one row")
    result = tuple(
        tuple(_rational(value, "matrix entry") for value in _sequence(row, "row"))
        for row in rows
    )
    width = len(result[0])
    if width == 0 or any(len(row) != width for row in result):
        raise ValueError("matrix must be nonempty and rectangular")
    return result


def _weights(
    values: Sequence[Rational] | None, size: int, name: str
) -> tuple[Fraction, ...]:
    if values is None:
        return (Fraction(1),) * size
    result = tuple(_rational(value, name) for value in _sequence(values, name))
    if len(result) != size or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain {size} strictly positive weights")
    return result


def _seal(payload: dict[str, Any], claim: str) -> dict[str, Any]:
    certificate = make_certificate(
        claim=claim,
        payload=payload,
        meta={
            "transcend_backend": "not_used",
            "proof_scope": "finite rational arithmetic; external analytical premises are not verified",
        },
    )
    return {**deepcopy(payload), "certificate": certificate}


def weighted_schur_bound(
    matrix: Sequence[Sequence[Rational]],
    row_weights: Sequence[Rational] | None = None,
    column_weights: Sequence[Rational] | None = None,
) -> dict[str, Any]:
    """Bound the l2 norm squared of D_row A D_column**(-1) by exact R*C.

    Omitted weights are all one. Row and column weights are norm multipliers,
    not probability weights. Signed entries are permitted; absolute values
    are taken after exact weighted similarity. The returned rational strings
    retain exact values and require no rounded square root.
    """
    entries = _matrix(matrix)
    n_rows, n_columns = len(entries), len(entries[0])
    rw = _weights(row_weights, n_rows, "row_weights")
    cw = _weights(column_weights, n_columns, "column_weights")
    transformed = tuple(
        tuple(abs(entries[i][j]) * rw[i] / cw[j] for j in range(n_columns))
        for i in range(n_rows)
    )
    row_sums = tuple(sum(row, Fraction(0)) for row in transformed)
    column_sums = tuple(
        sum((transformed[i][j] for i in range(n_rows)), Fraction(0))
        for j in range(n_columns)
    )
    row_cap, column_cap = max(row_sums), max(column_sums)
    payload = {
        "type": _SCHUR,
        "status": "PASS",
        "passed": True,
        "inputs": {
            "matrix": [[str(value) for value in row] for row in entries],
            "row_weights": [str(value) for value in rw],
            "column_weights": [str(value) for value in cw],
        },
        "bounds": {
            "row_sums": [str(value) for value in row_sums],
            "column_sums": [str(value) for value in column_sums],
            "row_upper": str(row_cap),
            "column_upper": str(column_cap),
            "l2_norm_squared_upper": str(row_cap * column_cap),
        },
        "norm": "Euclidean operator norm of diag(row_weights)*matrix*diag(column_weights)^(-1)",
        "external_premises": [],
        **_SCOPE,
    }
    return _seal(
        payload, "Exact rectangular weighted Schur bound for the supplied finite matrix"
    )


def _metric(
    distances: Sequence[Sequence[int]], size: int
) -> tuple[tuple[int, ...], ...]:
    rows = _sequence(distances, "distances")
    if len(rows) != size:
        raise ValueError("distance matrix must match the generator size")
    metric: list[tuple[int, ...]] = []
    for row in rows:
        values = tuple(_sequence(row, "distance row"))
        if len(values) != size:
            raise ValueError("distance matrix must be square")
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise TypeError("distances must be exact integers, not bools")
        metric.append(values)
    for i in range(size):
        for j in range(size):
            if (i == j and metric[i][j] != 0) or (i != j and metric[i][j] <= 0):
                raise ValueError(
                    "metric has zero diagonal and strictly positive off-diagonal distances"
                )
            if metric[i][j] != metric[j][i]:
                raise ValueError("distance metric must be symmetric")
            if any(metric[i][j] > metric[i][k] + metric[k][j] for k in range(size)):
                raise ValueError("distance matrix violates the triangle inequality")
    return tuple(metric)


def weighted_symmetric_generator_bounds(
    matrix: Sequence[Sequence[Rational]],
    distances: Sequence[Sequence[int]],
    weight: Rational,
) -> dict[str, Any]:
    """Compute exact finite coercivity, weighted decay, and Sylvester margins.

    A nonpositive Gershgorin floor is refused: it is not a proof that the
    matrix is indefinite. A positive floor with nonpositive weighted margin
    returns INCONCLUSIVE and no two-sided inverse bound. Metric validation
    is cubic in the finite matrix size. No infinite-size claim is inferred.
    """
    entries = _matrix(matrix)
    size = len(entries)
    if len(entries[0]) != size:
        raise ValueError("generator matrix must be square")
    if any(entries[i][j] != entries[j][i] for i in range(size) for j in range(size)):
        raise ValueError("generator matrix must be symmetric")
    metric = _metric(distances, size)
    base = _rational(weight, "weight")
    if base <= 1:
        raise ValueError("weight must be strictly greater than one")
    floors = tuple(
        entries[i][i]
        - sum((abs(entries[i][j]) for j in range(size) if i != j), Fraction(0))
        for i in range(size)
    )
    gap = min(floors)
    if gap <= 0:
        raise ValueError("a strictly positive Gershgorin coercivity floor is required")
    commutator_rows = tuple(
        sum(
            (
                abs(entries[i][j]) * (base ** metric[i][j] - 1)
                for j in range(size)
                if i != j
            ),
            Fraction(0),
        )
        for i in range(size)
    )
    eta = max(commutator_rows)
    margin = gap - eta
    two_sided, one_sided = 2 * margin, 2 * gap - eta
    passed = margin > 0
    payload = {
        "type": _GENERATOR,
        "status": "PASS" if passed else "INCONCLUSIVE",
        "passed": passed,
        "inputs": {
            "matrix": [[str(value) for value in row] for row in entries],
            "distances": [list(row) for row in metric],
            "weight": str(base),
        },
        "bounds": {
            "gershgorin_row_floors": [str(value) for value in floors],
            "gap_floor": str(gap),
            "weighted_commutator_row_uppers": [str(value) for value in commutator_rows],
            "weighted_commutator_upper": str(eta),
            "weighted_decay_margin": str(margin),
            "two_sided_hessian_decay_margin": str(two_sided),
            "one_sided_hessian_decay_margin": str(one_sided),
            "weighted_resolvent_upper": str(1 / margin) if margin > 0 else None,
            "two_sided_hessian_resolvent_upper": str(1 / two_sided)
            if two_sided > 0
            else None,
            "one_sided_hessian_resolvent_upper": str(1 / one_sided)
            if one_sided > 0
            else None,
        },
        "norm": "Euclidean vectors and Hilbert-Schmidt matrices; weights D_ii=weight**h_i for a 1-Lipschitz height h in the supplied metric",
        "external_premises": [
            "application height is 1-Lipschitz in the supplied metric",
            "any identification with a differential operator or mixed Hessian requires a separate proof",
        ],
        "finite_coercivity_verified": True,
        "finite_weighted_decay_verified": passed,
        **_SCOPE,
    }
    return _seal(
        payload,
        "Exact weighted finite symmetric-generator and two-sided Sylvester decay budgets",
    )


def quadratic_ball_budget(
    defect: Rational,
    linear_bound: Rational,
    bilinear_bound: Rational,
    radius: Rational,
) -> dict[str, Any]:
    """Check T(x)=L(r-B(x,x)) on ||x||<=radius from external norm bounds.

    defect bounds ||r||, linear_bound bounds ||L||, and bilinear_bound bounds
    ||B(x,y)||/(||x||||y||) in specified compatible spaces. The mapped-radius
    bound is c*(defect+b*radius**2), and the Lipschitz bound is 2*c*b*radius.
    The finite inequalities alone never certify those external hypotheses.
    """
    d, c, b, r = (
        _rational(value, name)
        for value, name in (
            (defect, "defect"),
            (linear_bound, "linear_bound"),
            (bilinear_bound, "bilinear_bound"),
            (radius, "radius"),
        )
    )
    if min(d, c, b) < 0 or r <= 0:
        raise ValueError(
            "defect and operator bounds must be nonnegative; radius must be positive"
        )
    mapped, contraction = c * (d + b * r * r), 2 * c * b * r
    self_map = r - mapped
    passed = self_map >= 0 and contraction < 1
    payload = {
        "type": _BALL,
        "status": "PASS" if passed else "INCONCLUSIVE",
        "passed": passed,
        "inputs": {
            "defect": str(d),
            "linear_bound": str(c),
            "bilinear_bound": str(b),
            "radius": str(r),
        },
        "bounds": {
            "mapped_radius_upper": str(mapped),
            "contraction_upper": str(contraction),
            "self_map_margin": str(self_map),
            "strict_contraction_margin": str(1 - contraction),
        },
        "external_premises": {
            "complete_real_or_complex_Banach_space_and_closed_ball": "UNVERIFIED",
            "forcing_defect_bound_in_the_declared_source_space": "UNVERIFIED",
            "linear_map_bound_between_the_declared_spaces": "UNVERIFIED",
            "bilinear_map_bound_into_that_same_source_space": "UNVERIFIED",
            "composition_is_the_target_fixed_point_map": "UNVERIFIED",
        },
        "nonlinear_existence_verified": False,
        **_SCOPE,
    }
    return _seal(
        payload,
        "Exact finite quadratic self-map and contraction gate with external Banach hypotheses",
    )


def _decode_rational(value: Any) -> Fraction:
    if not isinstance(value, str):
        raise TypeError("serialized rational inputs must be strings")
    return Fraction(value)


def replay_locality_certificate(certificate: Any) -> bool:
    """Recompute the entire certificate, including scope and all exact bounds.

    A valid digest alone is insufficient. Canonical rebuilding also rejects a
    resealed altered result, missing/extra fields, or noncanonical inputs.
    """
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        inputs = payload["inputs"]
        kind = payload["type"]
        if kind in (_SCHUR, _GENERATOR):
            matrix = [
                [_decode_rational(value) for value in row] for row in inputs["matrix"]
            ]
            if kind == _SCHUR:
                result = weighted_schur_bound(
                    matrix,
                    [_decode_rational(value) for value in inputs["row_weights"]],
                    [_decode_rational(value) for value in inputs["column_weights"]],
                )
            else:
                result = weighted_symmetric_generator_bounds(
                    matrix, inputs["distances"], _decode_rational(inputs["weight"])
                )
        elif kind == _BALL:
            result = quadratic_ball_budget(
                *(
                    _decode_rational(inputs[name])
                    for name in ("defect", "linear_bound", "bilinear_bound", "radius")
                )
            )
        else:
            return False
        return bool(certificate == result["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = [
    "quadratic_ball_budget",
    "replay_locality_certificate",
    "weighted_schur_bound",
    "weighted_symmetric_generator_bounds",
]
