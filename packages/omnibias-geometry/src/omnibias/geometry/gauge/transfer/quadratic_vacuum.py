# SPDX-License-Identifier: Apache-2.0
"""Exact Riccati residual bounds for a finite quadratic vacuum.

This is a coupled harmonic model. Identifying it as the quadratic term of
Wilson YM does not control the nonlinear, large-field or continuum remainder.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _rational

Matrix = Sequence[Sequence[int | Q]]


def _matrix(matrix: Matrix, name: str) -> list[list[Q]]:
    result = [[_rational(x, name) for x in row] for row in matrix]
    n = len(result)
    if not n or any(len(row) != n for row in result):
        raise ValueError(f"{name} must be a nonempty square matrix")
    if any(result[i][j] != result[j][i] for i in range(n) for j in range(n)):
        raise ValueError(f"{name} must be symmetric")
    return result


def quadratic_vacuum_root(
    stiffness: Matrix, root_witness: Matrix, *,
    frequency_lower: int | Q, scale: int | Q = 1,
) -> dict[str, Any]:
    """Certify sqrt(K) near a rational positive matrix P, without commutation.

    The exact row bound p<=lambda_min(P), residual r>=||K-P²|| and
    s²<=p²-r imply K>=s² I and ||sqrt(K)-P||<=r/(p+s). The inverse
    of X->sqrt(K)X+Xsqrt(K) has operator-norm bound 1/(2s).
    ``scale`` records the physical normalization; it proves no RG premise.
    Nonpositive or singular witnesses are refused by the sufficient gate.
    """
    k, p_matrix = _matrix(stiffness, "stiffness"), _matrix(root_witness, "root_witness")
    if len(k) != len(p_matrix):
        raise ValueError("stiffness and root_witness must have the same shape")
    s, sigma = _rational(frequency_lower, "frequency_lower"), _rational(scale, "scale")
    if s <= 0 or sigma <= 0:
        raise ValueError("frequency_lower and scale must be strictly positive")
    n = len(k)
    residual = [[k[i][j] - sum((p_matrix[i][t] * p_matrix[t][j] for t in range(n)), Q(0))
                 for j in range(n)] for i in range(n)]
    p = min(p_matrix[i][i] - sum((abs(p_matrix[i][j]) for j in range(n) if j != i), Q(0))
            for i in range(n))
    r = max(sum((abs(x) for x in row), Q(0)) for row in residual)
    slack = p * p - r - s * s
    passed = p > 0 and slack >= 0
    error = r / (p + s) if passed else None
    earned = {"finite_quadratic_root_verified": passed, "finite_quadratic_gap_verified": passed}
    scope = {
        "nonlinear_wilson_vacuum_verified": False,
        "nonlinear_remainder_verified": False,
        "uniform_in_volume_claim": False, "uniform_in_a_claim": False,
        "infinite_volume_claim": False, "continuum_claim": False,
        "yang_mills_claim": False, "yang_mills_mass_gap_claim": False,
    }
    witness = {
        "inputs": {
            "stiffness": [[str(x) for x in row] for row in k],
            "root_witness": [[str(x) for x in row] for row in p_matrix],
            "frequency_lower": str(s), "scale": str(sigma),
        },
        "model": "H_quad=kappa/2*(-Delta)+q^T K q/(2*kappa), q in R^n, kappa>0",
        "vacuum": "normalized exp(-q^T sqrt(K) q/(2*kappa))",
        "zero_modes": "must be removed by an explicitly justified physical coordinate restriction; no automatic quotient",
        "arithmetic": {
            "root_row_lower": str(p), "residual_matrix": [[str(x) for x in row] for row in residual],
            "residual_norm_upper": str(r), "frequency_squared_slack": str(slack),
            "root_error_upper": str(error) if error is not None else None,
            "normalized_root_error_upper": str(error / sigma) if error is not None else None,
            "quadratic_gap_lower": str(s) if passed else None,
            "riccati_inverse_upper": str(1 / (2 * s)) if passed else None,
            "normalized_riccati_inverse_upper": str(sigma / (2 * s)) if passed else None,
        },
        "proof": "Gershgorin; K=P^2+R>=s^2 I; Sylvester integral solves sqrt(K)D+DP=R",
        "failed_constraints": ([] if p > 0 else ["positive_root_row_lower"])
        + ([] if slack >= 0 else ["frequency_squared_slack"]),
    }
    cert = make_certificate(
        claim="finite rational residual bounds for the unique positive quadratic vacuum matrix and its Sylvester inverse",
        payload={"type": "quadratic_vacuum_root_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={"analytic_implication": "docs/api/gauge-quadratic-vacuum.md", "transcend_backend": "not_used"},
    )
    return {
        "status": "PASS" if passed else "INCONCLUSIVE", "witness": witness,
        "certificate": cert, "digest_verified": verify_certificate_digest(cert),
        **earned, **scope, "theorem_prover_verified": False, "mathlib_verified": False,
    }


def replay_quadratic_vacuum_root_certificate(certificate: dict[str, Any]) -> bool:
    """Replay all inputs, arithmetic, claims and scope, including refusals."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "quadratic_vacuum_root_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = quadratic_vacuum_root(
            [[Q(x) for x in row] for row in inputs["stiffness"]],
            [[Q(x) for x in row] for row in inputs["root_witness"]],
            frequency_lower=Q(inputs["frequency_lower"]), scale=Q(inputs["scale"]),
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = ["quadratic_vacuum_root", "replay_quadratic_vacuum_root_certificate"]
