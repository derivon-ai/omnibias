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


__all__ = [
    "gates_block",
    "mse",
    "rel_l2",
    "require_reference_valid",
    "require_rel_l2",
    "require_skill",
    "skill_score",
]
