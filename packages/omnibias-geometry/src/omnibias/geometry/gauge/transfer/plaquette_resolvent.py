# SPDX-License-Identifier: Apache-2.0
"""Full-spin inverse bounds for a one-plaquette Wilson-reference linearization.

The reference is explicit. The target Yang--Mills vacuum is not supplied or
constructed here. A finite rational block and a proved bound on every omitted
character certify the original Fourier norm; a different weighted Hilbert
norm has an unconditional inverse bound of one at every finite coupling.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational


def _energy(n: int) -> Q:
    return Q(n * (n + 2))


def _weight(n: int) -> Q:
    return Q(n) * _energy(n) * (n + 1) ** 3 / 2


def _up(t: Q, n: int) -> Q:
    return 2 * t * n / _energy(n + 1)


def _down(t: Q, n: int) -> Q:
    return 2 * t * (n + 2) / _energy(n - 1) if n > 1 else Q(0)


def _up_norm(t: Q, n: int) -> Q:
    return 2 * t * (n + 2) ** 2 / (n * (n + 1) ** 2)


def _down_norm(t: Q, n: int) -> Q:
    return 2 * t * n * (n - 1) / (n + 1) ** 3


def _inverse_columns(t: Q, size: int) -> tuple[list[Q], list[list[Q]], bool]:
    """Exact tridiagonal LU, including a direct inverse residual check."""
    pivots = [Q(1)]
    for n in range(2, size + 1):
        pivots.append(1 + _up(t, n - 1) * _down(t, n) / pivots[-1])
    columns: list[list[Q]] = []
    for column_index in range(size):
        rhs = [Q(int(i == column_index)) for i in range(size)]
        for i in range(1, size):
            rhs[i] -= _up(t, i) * rhs[i - 1] / pivots[i - 1]
        result = [Q(0)] * size
        result[-1] = rhs[-1] / pivots[-1]
        for i in reversed(range(size - 1)):
            result[i] = (rhs[i] + _down(t, i + 2) * result[i + 1]) / pivots[i]
        columns.append(result)
    residual_verified = True
    for j, column in enumerate(columns):
        for i, value in enumerate(column):
            if i > 0:
                value += _up(t, i) * column[i - 1]
            if i + 1 < size:
                value -= _down(t, i + 2) * column[i + 1]
            residual_verified &= value == int(i == j)
    return pivots, columns, residual_verified


def su2_plaquette_linearized_inverse(
    kappa: int | Q,
    *,
    cutoff: int = 16,
) -> dict[str, Any]:
    """Bound the inverse of I-2*C0^-1*Pi0*Gamma(S*, .) on all characters.

    S*=(g/3)*chi, g=4/kappa^2, and C chi_(n/2)=n(n+2) chi_(n/2).
    The positive cutoff retains nonconstant labels 1 through cutoff.
    Every omitted label is bounded analytically, not sampled. A failed
    Fourier defect gate is INCONCLUSIVE, while the separately labelled
    weighted-Hilbert inverse bound of one remains valid. No target-vacuum,
    physical-gap or volume/continuum claim follows from this reference inverse.
    """
    coupling = _rational(kappa, "kappa")
    size = _integer(cutoff, "cutoff")
    if coupling <= 0 or size <= 0:
        raise ValueError("kappa and cutoff must be strictly positive")
    g, t = 4 / coupling**2, Q(4, 3) / coupling**2
    pivots, columns, finite_verified = _inverse_columns(t, size)
    weighted_columns = [
        sum((_weight(i + 1) * abs(value) for i, value in enumerate(column)), Q(0)) / _weight(j + 1)
        for j, column in enumerate(columns)
    ]
    finite_norm = max(weighted_columns)
    preconditioner_norm = max(Q(1), finite_norm)
    retained_leak = _up_norm(t, size)
    boundary_column = _up_norm(t, size + 1) + _down_norm(t, size + 1) * weighted_columns[-1]
    far_tail = _up_norm(t, size + 2) + _down_norm(t, size + 2)
    defect = max(retained_leak, boundary_column, far_tail)
    passed = finite_verified and all(p > 0 for p in pivots) and defect < 1
    inverse = preconditioner_norm / (1 - defect) if passed else None
    # The Hilbert skew balance uses only exact squared weights, not square roots.
    skew_verified = all(
        _energy(n + 1) ** 2 * (n + 2) * _up(t, n) == _energy(n) ** 2 * (n + 1) * _down(t, n + 1)
        for n in range(1, size + 1)
    )
    earned = {
        "finite_reference_block_inverse_verified": finite_verified,
        "weighted_hilbert_reference_inverse_verified": skew_verified,
        "full_spin_fourier_reference_inverse_verified": passed,
        "reference_inverse_verified": passed,
    }
    scope = {
        "actual_vacuum_claim": False,
        "actual_vacuum_verified": False,
        "physical_mass_gap_claim": False,
        "uniform_in_volume_claim": False,
        "all_scale_refinement_claim": False,
        "infinite_volume_claim": False,
        "uniform_in_a_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
    }
    witness = {
        "inputs": {"kappa": str(coupling), "cutoff": size},
        "model": "one neutral SU2 plaquette, all character spins, four unit electric edges",
        "normalization": "aH=kappa/2*C+2/kappa*(2-chi), C=4*(-Delta_SU2), g=4/kappa^2",
        "reference": "S*=t*chi, t=g/3; explicit drift density exp(2*t*chi)*Haar/Z",
        "operator": "L=I-2*C0^-1*Pi0*Gamma(S*,.); C0^-1 and Pi0 use nonconstant Haar characters",
        "constant_mode": (
            "n=0 removed; solutions have zero Haar mean, not necessarily zero "
            "reference-density mean"
        ),
        "character_energy": "lambda_n=n*(n+2), n=2*j>=1",
        "operator_columns": "L chi_n=chi_n+a_n chi_(n+1)-b_n chi_(n-1), omitting n=0",
        "coefficient_up": "a_n=2*t*n/((n+1)*(n+3))",
        "coefficient_down": "b_n=2*t*(n+2)/((n-1)*(n+1)) for n>=2; b_1=0",
        "fourier_norm": (
            "sum_(n>=1) omega_n*abs(h_n), omega_n=n^2*(n+2)*(n+1)^3/2; canonical square orientation"
        ),
        "hilbert_norm": "sum_(n>=1) lambda_n^2*(n+1)*abs(h_n)^2, followed by square root",
        "hilbert_proof": (
            "D_n=lambda_n*sqrt(n+1); D*L*D^-1=I+J, J skew-adjoint with off-diagonal "
            "magnitude 2*t/sqrt((n+1)*(n+2))"
        ),
        "hilbert_inverse_upper": "1",
        "all_coupling_scope": (
            "the Hilbert inverse exists for every finite positive kappa; the Fourier "
            "inverse exists too by compactness/Fredholm, but this report earns its "
            "quantitative bound only through the finite defect gate"
        ),
        "tail_proof": (
            "weighted K=I-L column sum A_n+B_n decreases for all n>=1; retain 1..N and "
            "precondition by inverse(L_N) direct_sum I_tail"
        ),
        "weighted_up_formula": "A_n=2*t*(n+2)^2/(n*(n+1)^2)",
        "weighted_down_formula": "B_n=2*t*n*(n-1)/(n+1)^3",
        "column_monotonicity": (
            "(A_n+B_n)-(A_(n+1)+B_(n+1))=4*t*(n^5+6*n^4+22*n^3+47*n^2+47*n+16)/(n*(n+1)^3*(n+2)^3)>0"
        ),
        "arithmetic": {
            "g": str(g),
            "t": str(t),
            "cutoff": size,
            "generic_linear_norm_upper": str(32 * t),
            "exact_unpreconditioned_fourier_norm": str(Q(9, 2) * t),
            "retained_lu_pivots": [str(p) for p in pivots],
            "retained_inverse_weighted_column_norms": [str(v) for v in weighted_columns],
            "retained_inverse_norm": str(finite_norm),
            "preconditioner_norm": str(preconditioner_norm),
            "retained_to_omitted_defect": str(retained_leak),
            "first_omitted_column_defect": str(boundary_column),
            "all_further_omitted_columns_defect": str(far_tail),
            "all_spin_defect_norm": str(defect),
            "strict_defect_slack": str(1 - defect),
            "finite_inverse_residual_verified": finite_verified,
            "retained_skew_balance_verified": skew_verified,
            "fourier_inverse_upper": str(inverse) if passed else None,
        },
        "failed_constraints": [] if passed else ["strict_all_spin_preconditioned_defect"],
    }
    certificate = make_certificate(
        claim=(
            "the explicit one-plaquette SU2 reference linearization has a weighted Hilbert "
            "inverse bound and, when the exact finite block plus all-tail defect passes, a "
            "full-spin original-Fourier inverse bound"
        ),
        payload={"type": "su2_plaquette_linearized_inverse_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={
            "analytic_implication": "docs/api/gauge-plaquette-resolvent.md",
            "transcend_backend": "not_used",
        },
    )
    return {
        "status": "PASS" if passed else "INCONCLUSIVE",
        "finite_gate_verified": passed,
        "weighted_hilbert_inverse_upper": "1",
        "fourier_inverse_upper": str(inverse) if passed else None,
        "witness": witness,
        "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        **earned,
        **scope,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


def replay_su2_plaquette_linearized_inverse_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild the full canonical PASS or INCONCLUSIVE reference-inverse report."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_plaquette_linearized_inverse_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = su2_plaquette_linearized_inverse(Q(inputs["kappa"]), cutoff=inputs["cutoff"])
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_su2_plaquette_linearized_inverse_certificate",
    "su2_plaquette_linearized_inverse",
]
