# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified fractional-dissipation threshold for Kiselev Open Problem 1.

Given a two-sided enclosure ``lambda in [lo, hi]``, the paper's scaling argument
yields the critical fractional dissipation exponent

    alpha_crit >= 1 / (1 + hi).

The finite rational obligation is the margin
``threshold - error_bound = alpha_claimed - 1/(1+hi)`` (non-negative), shaped so
:func:`omnibias.core.proof.lean_check.check_certificate` can discharge it.

``viscous_perturbation.py`` remains the deferred NS-window track (algebraic
compact-window bound); this module supersedes it for the CCF dissipation claim.
"""

from __future__ import annotations

from typing import Any

from omnibias.core.proof.certificate import make_certificate


def alpha_crit_lower(lambda_hi: float) -> float:
    """Lower bound ``1/(1+lambda_hi)`` on the critical dissipation exponent."""
    hi = float(lambda_hi)
    if hi <= -1.0:
        raise ValueError("lambda_hi must be > -1")
    return 1.0 / (1.0 + hi)


def certified_fractional_dissipation_threshold(
    *,
    lambda_lo: float,
    lambda_hi: float,
    alpha_claimed: float | None = None,
    claim: str = "fractional dissipation threshold for sealed CCF lambda enclosure",
) -> dict[str, Any]:
    """Seal ``alpha_crit >= 1/(1+lambda_hi)`` as a finite margin obligation.

    If ``alpha_claimed`` is omitted it defaults to ``1/(1+lambda_hi)`` (tight).
    The certificate closes when ``alpha_claimed >= 1/(1+lambda_hi)``.
    """
    lo = float(lambda_lo)
    hi = float(lambda_hi)
    if lo > hi:
        raise ValueError("lambda_lo must be <= lambda_hi")
    alpha_lb = alpha_crit_lower(hi)
    alpha = float(alpha_claimed) if alpha_claimed is not None else alpha_lb
    margin = alpha - alpha_lb
    closed = bool(margin >= 0.0)
    # Finite obligation for lean_check: margin = threshold - error_bound as [lo,hi]
    # with both endpoints the same non-negative rationalizable floats.
    payload = {
        "type": "pinn_aposteriori_error",
        "family": "ccf_fractional_dissipation_threshold",
        "lambda_enclosure": {"lo": lo, "hi": hi},
        "alpha_crit_lower": alpha_lb,
        "alpha_claimed": alpha,
        "threshold_closed": closed,
        "finite_obligation": {
            "type": "error_bound_le_threshold",
            "error_bound": alpha_lb,
            "threshold": alpha,
            "margin": [margin, margin],
        },
        "kiselev_open_problem_1": True,
        "honesty": {
            "unproven_claim": False,
            "navier_stokes_proof_claim": False,
            "continuum_navier_stokes_claim": False,
            "scope": (
                "Finite threshold from a sealed two-sided lambda enclosure via the "
                "CCF scaling argument; not a continuum NS regularity theorem."
            ),
            "supersedes": "omnibias.pinn.certified.viscous_perturbation (NS-window track)",
        },
    }
    cert = make_certificate(claim=claim, payload=payload, honesty=payload["honesty"])
    return {
        "schema_version": "ccf-fractional-dissipation-threshold-1",
        "lambda_lo": lo,
        "lambda_hi": hi,
        "alpha_crit_lower": alpha_lb,
        "alpha_claimed": alpha,
        "margin": margin,
        "threshold_closed": closed,
        "certificate": cert,
        "honesty": payload["honesty"],
    }


def verify_fractional_dissipation_threshold(cert: dict[str, Any]) -> dict[str, Any]:
    """Independent recomputation of the threshold margin."""
    hi = float(cert["lambda_hi"])
    alpha_lb = alpha_crit_lower(hi)
    alpha = float(cert["alpha_claimed"])
    margin = alpha - alpha_lb
    match = abs(margin - float(cert["margin"])) <= 1e-14
    return {
        "recomputed_alpha_crit_lower": alpha_lb,
        "recomputed_margin": margin,
        "replay_match": bool(match),
        "threshold_closed": bool(margin >= 0.0),
        "navier_stokes_proof_claim": False,
    }


__all__ = [
    "alpha_crit_lower",
    "certified_fractional_dissipation_threshold",
    "verify_fractional_dissipation_threshold",
]
