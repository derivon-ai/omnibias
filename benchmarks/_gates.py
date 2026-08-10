# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Absolute, physics-anchored acceptance gates for the public benchmarks.

Relative comparisons between two failing arms, or existential checks like
``isfinite``, are not gates. A gate must answer three questions in order:

1. Is the *reference* physically valid?
2. Does the prediction beat the trivial zero predictor (skill > 0)?
3. Does the absolute error clear a named threshold?

Every artifact that imports this module should emit a ``gates`` block so a
JSON can never again look like a result while encoding a divergence.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def mse(pred: Any, target: Any) -> float:
    """Mean squared error between array-likes."""
    p = np.asarray(pred, dtype=float).reshape(-1)
    t = np.asarray(target, dtype=float).reshape(-1)
    if p.shape != t.shape:
        raise ValueError(f"mse shape mismatch: {p.shape} vs {t.shape}")
    return float(np.mean((p - t) ** 2))


def rel_l2(pred: Any, target: Any, *, eps: float = 1e-30) -> float:
    """Relative L2 error ``||pred - target||_2 / ||target||_2``."""
    p = np.asarray(pred, dtype=float).reshape(-1)
    t = np.asarray(target, dtype=float).reshape(-1)
    if p.shape != t.shape:
        raise ValueError(f"rel_l2 shape mismatch: {p.shape} vs {t.shape}")
    denom = float(np.linalg.norm(t))
    if denom < eps:
        raise ValueError("rel_l2: target has near-zero norm; pick a nontrivial field")
    return float(np.linalg.norm(p - t) / denom)


def skill_score(pred: Any, target: Any, *, eps: float = 1e-30) -> float:
    """Nash-Sutcliffe skill: ``1 - MSE(pred)/MSE(0)``.

    Positive skill means the prediction beats the trivial zero predictor.
    A score of 1.0 is a perfect match; <= 0 means worse than predicting zero.
    """
    p = np.asarray(pred, dtype=float).reshape(-1)
    t = np.asarray(target, dtype=float).reshape(-1)
    mse_pred = float(np.mean((p - t) ** 2))
    mse_zero = float(np.mean(t**2))
    if mse_zero < eps:
        raise ValueError("skill_score: target energy near zero")
    return 1.0 - mse_pred / mse_zero


def require_reference_valid(
    values: Any,
    *,
    u0_max_abs: float | None = None,
    max_abs_cap: float | None = None,
    name: str = "reference",
) -> dict[str, Any]:
    """Assert a physical validity floor on reference data.

    For the periodic heat equation the maximum principle implies
    ``max|u(t)| <= max|u(0)|``. Passing ``u0_max_abs`` enforces that.
    ``max_abs_cap`` is a hard absolute ceiling (e.g. for manufactured
    solutions whose amplitude is known a priori).
    """
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name}: non-finite values (INVALID EXPERIMENT)")
    max_abs = float(np.max(np.abs(arr)))
    verdict: dict[str, Any] = {
        "name": name,
        "max_abs": max_abs,
        "finite": True,
        "passed": True,
    }
    if u0_max_abs is not None:
        # Allow a tiny float cushion; a genuine blow-up is orders of magnitude.
        ok = max_abs <= float(u0_max_abs) * (1.0 + 1e-6) + 1e-12
        verdict["u0_max_abs"] = float(u0_max_abs)
        verdict["maximum_principle"] = ok
        if not ok:
            raise RuntimeError(
                f"{name}: maximum principle violated "
                f"(max|u|={max_abs:.3e} > max|u0|={u0_max_abs:.3e}); "
                "INVALID EXPERIMENT -- fix the reference integrator"
            )
    if max_abs_cap is not None:
        ok = max_abs <= float(max_abs_cap)
        verdict["max_abs_cap"] = float(max_abs_cap)
        verdict["within_cap"] = ok
        if not ok:
            raise RuntimeError(
                f"{name}: max|u|={max_abs:.3e} exceeds cap {max_abs_cap:.3e}; "
                "INVALID EXPERIMENT"
            )
    return verdict


def require_skill(
    pred: Any,
    target: Any,
    *,
    min_skill: float = 0.0,
    name: str = "prediction",
) -> dict[str, Any]:
    """Require ``skill_score >= min_skill`` (default: beat the zero predictor)."""
    skill = skill_score(pred, target)
    passed = skill >= min_skill
    verdict = {
        "name": name,
        "skill_score": skill,
        "min_skill": min_skill,
        "passed": passed,
    }
    if not passed:
        raise AssertionError(
            f"{name}: skill_score={skill:.4f} < {min_skill} "
            "(does not beat the zero predictor -- not a solved experiment)"
        )
    return verdict


def require_rel_l2(
    pred: Any,
    target: Any,
    *,
    max_rel_l2: float,
    name: str = "prediction",
) -> dict[str, Any]:
    """Require absolute relative-L2 below a named threshold."""
    err = rel_l2(pred, target)
    passed = err <= max_rel_l2
    verdict = {
        "name": name,
        "rel_l2": err,
        "max_rel_l2": max_rel_l2,
        "passed": passed,
    }
    if not passed:
        raise AssertionError(
            f"{name}: rel_l2={err:.4e} exceeds gate {max_rel_l2:.4e}"
        )
    return verdict


def gates_block(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-gate verdicts into the artifact ``gates`` block."""
    # Coerce numpy scalars so json.dumps never sees np.bool_.
    clean: list[dict[str, Any]] = []
    for e in entries:
        item = dict(e)
        if "passed" in item:
            item["passed"] = bool(item["passed"])
        clean.append(item)
    return {
        "all_passed": all(bool(e.get("passed", False)) for e in clean),
        "entries": clean,
    }


# Published CCF self-similar lambda digits (DeepMind arXiv:2509.14185).
CCF_LAMBDA_1ST_UNSTABLE = 0.6057
CCF_LAMBDA_2ND_UNSTABLE = 0.4703
CCF_RESIDUAL_GATE_STABLE = 1e-11
CCF_RESIDUAL_GATE_1ST_UNSTABLE = 1e-11
CCF_RESIDUAL_GATE_2ND_UNSTABLE = 1e-6
CCF_STRETCH_RESIDUAL_GATE = 1e-13


def ccf_lambda_digits_gate(
    lam: float,
    target: float,
    *,
    abs_tol: float = 5e-5,
    name: str = "ccf_lambda",
) -> dict[str, Any]:
    """Require recovered ``lambda`` to match a published target within ``abs_tol``."""
    err = abs(float(lam) - float(target))
    passed = bool(err <= float(abs_tol))
    return {
        "name": name,
        "lam": float(lam),
        "target": float(target),
        "abs_error": err,
        "abs_tol": float(abs_tol),
        "passed": passed,
    }


def ccf_residual_gate(
    max_abs_residual: float,
    threshold: float,
    *,
    name: str = "ccf_residual",
) -> dict[str, Any]:
    """Require whole-line max |d0 residual| at or below ``threshold``."""
    val = float(max_abs_residual)
    thr = float(threshold)
    passed = bool(np.isfinite(val) and val <= thr)
    return {
        "name": name,
        "max_abs_residual": val,
        "threshold": thr,
        "passed": passed,
    }


def ccf_absolute_gates(
    *,
    lam: float,
    max_abs_residual: float,
    family: str = "1st_unstable",
    stretch_mp_residual: float | None = None,
) -> dict[str, Any]:
    """Absolute Rung-1 gates for a CCF Hardy discovery result.

    ``family`` is one of ``stable``, ``1st_unstable``, ``2nd_unstable``.
    Empirical-law outputs must never be passed as ``target`` — callers supply
    published digits only.
    """
    if family == "1st_unstable":
        target = CCF_LAMBDA_1ST_UNSTABLE
        res_thr = CCF_RESIDUAL_GATE_1ST_UNSTABLE
    elif family == "2nd_unstable":
        target = CCF_LAMBDA_2ND_UNSTABLE
        res_thr = CCF_RESIDUAL_GATE_2ND_UNSTABLE
    elif family == "stable":
        target = CCF_LAMBDA_1ST_UNSTABLE  # placeholder: stable digits vary by paper
        res_thr = CCF_RESIDUAL_GATE_STABLE
    else:
        raise ValueError(f"unknown CCF family {family!r}")
    entries = [
        ccf_lambda_digits_gate(lam, target, name=f"lambda_{family}"),
        ccf_residual_gate(max_abs_residual, res_thr, name=f"residual_{family}"),
    ]
    if stretch_mp_residual is not None:
        entries.append(
            ccf_residual_gate(
                stretch_mp_residual,
                CCF_STRETCH_RESIDUAL_GATE,
                name="stretch_mpmath_residual",
            )
        )
    block = gates_block(entries)
    residual_ok = bool(entries[1]["passed"])
    lambda_ok = bool(entries[0]["passed"])
    return {
        "family": family,
        "earned": bool(block["all_passed"]),
        "gates": block,
        "honesty": {
            # Both residual and lambda digits must clear — lambda alone (e.g. frozen init) is not reproduction.
            "reproduces_published_lambda": bool(lambda_ok and residual_ok),
            "navier_stokes_proof_claim": False,
            "anti_circularity": "targets are published digits, never empirical-law outputs",
        },
    }


# IPM / Boussinesq scaffold residual floors (promote when discovery upgrades).
# Gaussian Adam smoke clears ~O(1); tighten when CubicGN/Martens paths land.
IPM_SCAFFOLD_RESIDUAL_GATE = 2.0
BOUSSINESQ_SCAFFOLD_RESIDUAL_GATE = 2.0


def ipm_boussinesq_scaffold_gates(
    *,
    family: str,
    max_abs_residual: float,
    navier_stokes_proof_claim: bool = False,
) -> dict[str, Any]:
    """Scaffold absolute gate for IPM/Boussinesq smoke (not Clay NS).

    ``earned`` stays False while residual exceeds the scaffold floor or if any
    continuum NS claim is asserted. Thresholds tighten when discovery leaves
    Gaussian Adam smoke.
    """
    if navier_stokes_proof_claim:
        return {
            "family": family,
            "earned": False,
            "gates": gates_block(
                [
                    {
                        "name": "honesty_ns_claim_blocked",
                        "passed": False,
                        "detail": "navier_stokes_proof_claim must be False",
                    }
                ]
            ),
            "honesty": {
                "navier_stokes_proof_claim": False,
                "scaffold": True,
            },
        }
    thr = (
        IPM_SCAFFOLD_RESIDUAL_GATE
        if family == "ipm"
        else BOUSSINESQ_SCAFFOLD_RESIDUAL_GATE
    )
    entry = ccf_residual_gate(
        max_abs_residual, thr, name=f"{family}_scaffold_residual"
    )
    block = gates_block([entry])
    return {
        "family": family,
        "earned": bool(block["all_passed"]),
        "gates": block,
        "honesty": {
            "navier_stokes_proof_claim": False,
            "scaffold": True,
            "not_navier_stokes": True,
        },
    }


__all__ = [
    "CCF_LAMBDA_1ST_UNSTABLE",
    "CCF_LAMBDA_2ND_UNSTABLE",
    "CCF_RESIDUAL_GATE_1ST_UNSTABLE",
    "CCF_RESIDUAL_GATE_2ND_UNSTABLE",
    "CCF_RESIDUAL_GATE_STABLE",
    "CCF_STRETCH_RESIDUAL_GATE",
    "IPM_SCAFFOLD_RESIDUAL_GATE",
    "BOUSSINESQ_SCAFFOLD_RESIDUAL_GATE",
    "ccf_absolute_gates",
    "ccf_lambda_digits_gate",
    "ccf_residual_gate",
    "gates_block",
    "ipm_boussinesq_scaffold_gates",
    "mse",
    "rel_l2",
    "require_reference_valid",
    "require_rel_l2",
    "require_skill",
    "skill_score",
]
