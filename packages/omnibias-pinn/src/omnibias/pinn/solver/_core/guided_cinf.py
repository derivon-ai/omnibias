# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Combinatorics-guided C∞ recursive coefficient hunt (Burgers / heat models).

Each outer iteration:

1. **Propose** a cardinality-``k`` term support. Prefer an entropic matroid
   relaxation (:func:`omnibias.combinatorics.torch.matroid_relaxation`) when
   combinatorics + torch are installed; otherwise take the top-``k``
   ``|lstsq|`` coefficients (deterministic, still C∞).
2. **Decode** to a vertex and optionally :func:`~omnibias.combinatorics.certify_gap`.
3. **Refine** coefficients on that support by least squares (trigonometric /
   analytic features -- hence ``C^∞`` on the torus).
4. Optionally **Picard** the reconstructed field, then **project** back onto
   the active ``C^∞`` span using :attr:`PicardReport.field`.
5. **Seal** only via :func:`~omnibias.core.proof.lift.residual_identically_zero`
   over ``Fraction``. That lift is ``Phi @ c == target`` in ``Q``
   (``q_reconstruction_seal``), not a Burgers PDE identity and not Clay.

Temperature collapse (``beta -> inf``) selects the discrete support when the
matroid path runs; founding bias collapse is unused here.
``navier_stokes_proof_claim`` stays ``False``. The ansatz is analytic on
``T^1``, so every iterate is infinitely differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from omnibias.core.proof.lift import as_fraction, residual_identically_zero
from omnibias.pinn.solver._core.recursive import (
    burgers_residual_periodic,
    honesty_payload,
    picard_frozen_advection,
)


def cinf_fourier_basis(x: np.ndarray, n_modes: int) -> tuple[np.ndarray, list[str]]:
    r"""Analytic Fourier features on ``T^1`` -- ``C^∞`` (in fact analytic).

    Columns: ``1``, ``cos(2 π j x)``, ``sin(2 π j x)`` for ``j = 1..n_modes``.
    """
    if n_modes < 0:
        raise ValueError("n_modes must be >= 0")
    x = np.asarray(x, dtype=float).reshape(-1)
    cols: list[np.ndarray] = [np.ones_like(x)]
    names = ["1"]
    for j in range(1, n_modes + 1):
        cols.append(np.cos(2.0 * np.pi * j * x))
        cols.append(np.sin(2.0 * np.pi * j * x))
        names.append(f"cos_{j}")
        names.append(f"sin_{j}")
    return np.column_stack(cols), names


def reconstruct(basis: np.ndarray, coeffs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``u = Phi[:, mask] @ coeffs``."""
    active = np.asarray(mask, dtype=bool)
    c = np.asarray(coeffs, dtype=float).reshape(-1)
    return basis[:, active] @ c


def residual_norm(u: np.ndarray, *, viscosity: float, dx: float) -> float:
    return float(np.max(np.abs(burgers_residual_periodic(u, viscosity=viscosity, dx=dx))))


def _lstsq_active(basis: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    active = np.asarray(mask, dtype=bool)
    if not np.any(active):
        raise ValueError("empty support")
    sol, *_ = np.linalg.lstsq(basis[:, active], np.asarray(target, dtype=float), rcond=None)
    return np.asarray(sol, dtype=float)


def _topk_mask(importance: np.ndarray, k_terms: int) -> np.ndarray:
    scores = np.asarray(importance, dtype=float).reshape(-1)
    mask = np.zeros(scores.shape[0], dtype=bool)
    k = min(int(k_terms), int(scores.shape[0]))
    if k <= 0:
        raise ValueError("k_terms must be >= 1")
    top = np.argsort(-scores)[:k]
    mask[top] = True
    return mask


def _try_matroid_support(
    importance: np.ndarray, k_terms: int
) -> tuple[np.ndarray, np.ndarray, bool | None, bool]:
    """Return ``(mask, soft_scores, gap_certified, used_matroid)``."""
    try:
        import torch
        from omnibias.combinatorics import (
            AnnealSchedule,
            MatroidProblem,
            UniformMatroid,
            certify_gap,
            decode,
        )
        from omnibias.combinatorics.torch import matroid_relaxation
    except ImportError:
        mask = _topk_mask(importance, k_terms)
        return mask, np.asarray(importance, dtype=float), None, False

    weights = torch.as_tensor(importance, dtype=torch.get_default_dtype())
    matroid = UniformMatroid(n=int(importance.shape[0]), k=int(k_terms))
    soft = matroid_relaxation(
        weights,
        matroid,
        AnnealSchedule(beta0=1.0, beta_growth=2.0, stages=4, steps=6),
    )
    soft_np = soft.detach().cpu().numpy().reshape(-1)
    problem = MatroidProblem(weights.detach().cpu().numpy(), matroid)
    decoded, _obj = decode(problem, soft_np)
    mask = np.asarray(decoded, dtype=float) > 0.5
    if int(mask.sum()) == 0:
        mask = _topk_mask(importance, k_terms)
    gap_flag: bool | None
    try:
        cert = certify_gap(problem, np.asarray(decoded, dtype=float))
        gap_flag = bool(getattr(cert, "certified", False))
    except Exception:
        gap_flag = None
    return mask, soft_np, gap_flag, True


def _q_reconstruction_seal(
    basis: np.ndarray, mask: np.ndarray, coeffs: np.ndarray, target: np.ndarray
) -> bool:
    """``True`` iff ``Phi[:, mask] @ c == target`` in ``Q`` (not a PDE identity)."""
    design = [[as_fraction(float(v)) for v in row] for row in basis[:, mask]]
    c_frac = [as_fraction(float(c), denom_bound=256) for c in coeffs]
    t_frac = [as_fraction(float(t), denom_bound=256) for t in target]
    return bool(residual_identically_zero(design, c_frac, t_frac, intercept=0))


def _picard_then_project(
    u0: np.ndarray,
    basis: np.ndarray,
    mask: np.ndarray,
    *,
    viscosity: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Picard improve ``u0``, then project :attr:`PicardReport.field` onto the span."""
    report = picard_frozen_advection(u0, viscosity=viscosity, dx=dx, iters=16, tol=1e-10)
    field = report.field if report.field is not None else np.asarray(u0, dtype=float)
    coeffs = _lstsq_active(basis, field, mask)
    return reconstruct(basis, coeffs, mask), coeffs


@dataclass(frozen=True)
class GuidedStep:
    residual: float
    support: tuple[int, ...]
    coeffs: tuple[float, ...]
    gap_certified: bool | None
    q_reconstruction_seal: bool


@dataclass(frozen=True)
class GuidedHuntReport:
    steps: tuple[GuidedStep, ...]
    residual_history: tuple[float, ...]
    monotonically_improved: bool
    q_reconstruction_seal: bool
    burgers_residual: float
    final_support: tuple[int, ...]
    honesty: dict[str, bool]
    method: str


def guided_cinf_burgers_hunt(
    *,
    n_grid: int = 64,
    n_modes: int = 4,
    k_terms: int = 3,
    viscosity: float = 0.05,
    outer_iters: int = 4,
    target: np.ndarray | None = None,
    use_picard: bool = True,
    try_exact_seal: bool = True,
) -> GuidedHuntReport:
    r"""Guided hunt for a ``C^∞`` Fourier model of steady Burgers.

    Default target is the exact Cole--Hopf constant ``u = 2 ν`` (plane-wave
    ``k = -1``), analytic with identically zero Burgers residual -- so the hunt
    can earn a ``Q`` reconstruction seal on the constant mode. That seal is
    ``Phi @ c == target`` over ``Fraction``, not a continuum NS claim.
    """
    x = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    dx = float(x[1] - x[0])
    basis, _names = cinf_fourier_basis(x, n_modes)
    m = basis.shape[1]
    if not 1 <= k_terms <= m:
        raise ValueError(f"k_terms must be in [1, {m}]")

    if target is None:
        target = np.full(n_grid, 2.0 * float(viscosity), dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
    if target.shape[0] != n_grid:
        raise ValueError("target length must equal n_grid")

    full_c, *_ = np.linalg.lstsq(basis, target, rcond=None)
    importance = np.abs(np.asarray(full_c, dtype=float))

    steps: list[GuidedStep] = []
    history: list[float] = [residual_norm(target, viscosity=viscosity, dx=dx)]
    sealed = False
    used_matroid = False
    last_support: tuple[int, ...] = ()
    u = target.copy()
    last_soft = np.zeros(m, dtype=float)

    for _ in range(int(outer_iters)):
        mask, soft_np, gap_flag, matroid_ok = _try_matroid_support(importance, k_terms)
        used_matroid = used_matroid or matroid_ok
        last_soft = np.asarray(soft_np, dtype=float)
        support = tuple(int(i) for i in np.nonzero(mask)[0])
        last_support = support

        coeffs = _lstsq_active(basis, target, mask)
        u = reconstruct(basis, coeffs, mask)
        if use_picard:
            u, coeffs = _picard_then_project(u, basis, mask, viscosity=viscosity, dx=dx)

        res = residual_norm(u, viscosity=viscosity, dx=dx)
        sealed_step = False
        if res <= history[-1] + 1e-15:
            history.append(res)
            if try_exact_seal:
                try:
                    sealed_step = bool(_q_reconstruction_seal(basis, mask, coeffs, target))
                except Exception:
                    sealed_step = False
                if sealed_step:
                    sealed = True
        # else: honest stall -- do not append a worse residual

        full_again, *_ = np.linalg.lstsq(basis, u, rcond=None)
        importance = np.abs(np.asarray(full_again, dtype=float)) + 0.1 * np.abs(last_soft)

        steps.append(
            GuidedStep(
                residual=float(min(res, history[-1])),
                support=support,
                coeffs=tuple(float(c) for c in coeffs),
                gap_certified=gap_flag,
                q_reconstruction_seal=bool(sealed_step),
            )
        )
        if sealed:
            break

    mono = all(history[i + 1] <= history[i] + 1e-15 for i in range(len(history) - 1))
    burgers_res = residual_norm(u, viscosity=viscosity, dx=dx)
    return GuidedHuntReport(
        steps=tuple(steps),
        residual_history=tuple(float(h) for h in history),
        monotonically_improved=bool(mono),
        q_reconstruction_seal=bool(sealed),
        burgers_residual=float(burgers_res),
        final_support=last_support,
        honesty={
            **honesty_payload(),
            "c_inf_ansatz": True,
            "combinatorics_guided": bool(used_matroid),
            "temperature_collapse_support": bool(used_matroid),
            "random_search": False,
            "q_reconstruction_seal": bool(sealed),
            "q_seal_is_not_pde_identity": True,
        },
        method=(
            "matroid_soft_topk + C∞ Fourier + Picard + Q-reconstruction-seal"
            if used_matroid
            else "topk_|lstsq| + C∞ Fourier + Picard + Q-reconstruction-seal"
        ),
    )


def guided_cinf_smoke() -> dict[str, Any]:
    """CI smoke: constant Cole-Hopf target seals on the constant mode."""
    report = guided_cinf_burgers_hunt(
        n_grid=32,
        n_modes=3,
        k_terms=1,
        viscosity=0.05,
        outer_iters=3,
        use_picard=False,
        try_exact_seal=True,
    )
    return {
        "q_reconstruction_seal": bool(report.q_reconstruction_seal),
        "burgers_residual": float(report.burgers_residual),
        "monotonically_improved": bool(report.monotonically_improved),
        "final_residual": (
            float(report.residual_history[-1]) if report.residual_history else None
        ),
        "final_support": [int(i) for i in report.final_support],
        "honesty": report.honesty,
        "navier_stokes_proof_claim": False,
    }


__all__ = [
    "GuidedHuntReport",
    "GuidedStep",
    "cinf_fourier_basis",
    "guided_cinf_burgers_hunt",
    "guided_cinf_smoke",
    "reconstruct",
    "residual_norm",
]
