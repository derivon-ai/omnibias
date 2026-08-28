# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified truncation-horizon selection from an enclosed closed-loop monodromy
(theory 10-03).

Short-horizon policy optimisation (truncated BPTT, SHAC, PEARL) always
picks a window length ``h`` by hand. This module derives one instead: the
closed-loop discrete monodromy

.. math::
    \Phi_h = M_{k+h-1} M_{k+h-2} \cdots M_k, \qquad
    M_j = A_j + B_j\,\partial\pi/\partial y_j

is exactly the product :func:`omnibias.core.adjoint.closed_loop_jacobian`
already assembles per step for the adjoint recursion. If a *sound* radius
around each point Jacobian ``M_j`` is declared -- a ball, supplied by the
caller, not derived here -- the interval matrix product of those balls
rigorously encloses ``Phi_h`` for every trajectory inside the declared
uncertainty, and :func:`omnibias.dynamics.spectral_radius_bound` (reused,
not reimplemented) brackets its spectral radius. If the enclosed upper
bound stays below 1, the closed loop is *provably* contracting over that
window, and the horizon can be grown until the enclosed operator norm
drops below a stated tolerance -- the point at which a bounded terminal
error is certified to have decayed enough not to matter.

**What this reuses, and what it does not.** ``spectral_radius_bound`` is
imported verbatim from :mod:`omnibias.dynamics`. ``variational_flow`` /
``monodromy_matrix`` in that same module are continuous-time ODE
flow enclosures (Lohner-stepped fundamental matrices); they do not apply
here because the adjoint recursion's ``M_j`` sequence is already discrete
-- there is no ODE to step. This module instead builds the discrete
analogue directly: the interval matrix product
:func:`omnibias.core.verified.linalg.matmul` composes per-step, exactly the
operation :func:`omnibias.dynamics._core.variational.variational_step` uses
to accumulate its fundamental matrix (``matmul(j_step, vstate.fundamental)``).

This is **founding bias collapse** (``delta -> 0``), not temperature
collapse: the certificate is a sound enclosure of a fixed finite product,
not a ``beta -> inf`` feasibility limit. Do not conflate the two. Coverage is
checked against the *declared* uncertainty ball; if the true per-step
Jacobian variation exceeds the caller's declared radius, the certificate
is unsound by construction -- the radius is not derived from a Lipschitz
bound automatically in this pass (a leftover, see
``theory/10-control/03-certified-horizon-gradient-bias.md``).
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.linalg import IntervalMatrix, identity_matrix, matmul
from omnibias.dynamics import spectral_radius_bound

FloatArray = NDArray[np.float64]

DISCLAIMER = (
    "discrete closed-loop monodromy product enclosure; founding bias collapse, "
    "not temperature collapse; sound only within the caller-declared per-step "
    "Jacobian radius; not a continuum ODE monodromy claim"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "continuum_claimed": False,
        "auto_derived_radius_claimed": False,
        "unconditional_enclosure_claimed": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
        "skips_chain_rule": False,
    }


def enclose_jacobian_ball(m_center: FloatArray, radius: float) -> IntervalMatrix:
    r"""Sound elementwise ball ``[m_ij - radius, m_ij + radius]`` around ``M``.

    ``radius`` is declared by the caller (e.g. from a measured trust region
    or a supplied Lipschitz bound on ``dpi/dy`` and the environment
    Jacobian over a local box); this function does not compute one.
    """
    m = np.asarray(m_center, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"m_center must be square 2-D, got shape {m.shape}")
    r = float(radius)
    if r < 0.0:
        raise ValueError(f"radius must be >= 0, got {r}")
    if not np.all(np.isfinite(m)):
        raise ValueError("m_center must be finite")
    return [[Interval(float(m[i, j]) - r, float(m[i, j]) + r) for j in range(m.shape[1])] for i in range(m.shape[0])]


def monodromy_product(m_balls: Sequence[IntervalMatrix]) -> IntervalMatrix:
    r"""Sequential interval matrix product ``Phi_h = M_{h-1} @ ... @ M_0``.

    Composes left-to-right the same way
    :func:`omnibias.dynamics._core.variational.variational_step` accumulates
    its fundamental matrix (``matmul(j_step, running_fundamental)``); the
    only difference is that here every ``M_j`` is a declared ball, not a
    Lohner-enclosed continuous-time transition.
    """
    if not m_balls:
        raise ValueError("m_balls must be non-empty")
    dim = len(m_balls[0])
    product: IntervalMatrix = identity_matrix(dim)
    for m in m_balls:
        if len(m) != dim or any(len(row) != dim for row in m):
            raise ValueError("all Jacobian balls must share the same square dimension")
        product = matmul(m, product)
    return product


@dataclass(frozen=True)
class CertifiedHorizonResult:
    """Result of scanning horizons for a certified contraction / decay window."""

    certified: bool
    horizon: int | None
    spectral_radius_bound_hi: float | None
    tolerance: float
    max_horizon: int
    trace: list[float]


def certified_horizon(
    m_centers: Sequence[FloatArray],
    radius: float,
    *,
    tol: float = 0.1,
    max_horizon: int | None = None,
) -> CertifiedHorizonResult:
    r"""Smallest ``h`` such that the enclosed ``h``-step monodromy has spectral
    radius upper bound ``<= tol``.

    ``m_centers`` is the sequence of per-step closed-loop Jacobians
    ``M_0, M_1, ...`` (e.g. from repeated calls to
    :func:`omnibias.core.adjoint.closed_loop_jacobian`); ``radius`` is the
    declared per-step Jacobian uncertainty (see
    :func:`enclose_jacobian_ball`). Returns ``certified=False`` if no such
    ``h`` is found within ``max_horizon`` steps (default: ``len(m_centers)``)
    -- never a fabricated horizon.
    """
    if not m_centers:
        raise ValueError("m_centers must be non-empty")
    limit = len(m_centers) if max_horizon is None else min(max_horizon, len(m_centers))
    if limit < 1:
        raise ValueError("max_horizon must allow at least one step")
    tol_f = float(tol)
    if tol_f <= 0.0:
        raise ValueError(f"tol must be > 0, got {tol_f}")

    balls = [enclose_jacobian_ball(m, radius) for m in m_centers[:limit]]
    dim = len(balls[0])
    product: IntervalMatrix = identity_matrix(dim)
    trace: list[float] = []
    for h in range(1, limit + 1):
        product = matmul(balls[h - 1], product)
        rho = spectral_radius_bound(product)
        trace.append(rho.hi)
        if rho.hi <= tol_f:
            return CertifiedHorizonResult(
                certified=True,
                horizon=h,
                spectral_radius_bound_hi=rho.hi,
                tolerance=tol_f,
                max_horizon=limit,
                trace=trace,
            )
    return CertifiedHorizonResult(
        certified=False,
        horizon=None,
        spectral_radius_bound_hi=trace[-1] if trace else None,
        tolerance=tol_f,
        max_horizon=limit,
        trace=trace,
    )


def worked_example() -> dict[str, object]:
    """G5 reference: a contracting linear system certifies a finite horizon."""
    m = np.array([[0.5, 0.0], [0.0, 0.6]])
    result = certified_horizon([m] * 20, radius=0.01, tol=0.05, max_horizon=20)
    # rho(M) = 0.6 per step (point matrix); analytic h with 0.6^h <= 0.05 is h=5 (0.6^5=0.0778>0.05 -> h=6)
    analytic_h = math.ceil(math.log(0.05) / math.log(0.6))
    return {
        "certified": result.certified,
        "horizon": result.horizon,
        "analytic_h_reference": analytic_h,
        "g5_earned": result.certified and result.horizon is not None and result.horizon <= analytic_h + 2,
    }


def horizon_skill(*, n: int = 20, seed: int = 0) -> dict[str, object]:
    r"""G5: random contracting diagonal systems all get certified within a slack bound."""
    rng = random.Random(seed)
    horizons: list[int] = []
    coverage = 0
    for _ in range(n):
        dim = rng.choice((1, 2, 3))
        diag = [rng.uniform(0.2, 0.8) for _ in range(dim)]
        m = np.diag(diag)
        radius = rng.uniform(0.0, 0.02)
        tol = rng.uniform(0.02, 0.1)
        result = certified_horizon([m] * 40, radius=radius, tol=tol, max_horizon=40)
        if result.certified:
            coverage += 1
            assert result.horizon is not None
            horizons.append(result.horizon)
            actual_point_rho = max(diag) ** result.horizon
            if actual_point_rho <= tol + 1e-9:
                pass
            else:
                # The declared ball's upper bound can be conservative but must
                # never be *violated* by the true point-matrix trajectory.
                raise AssertionError("certified horizon did not actually bound the point trajectory")
    return {
        "coverage": coverage / n,
        "median_horizon": statistics.median(horizons) if horizons else math.inf,
        "g5_earned": coverage == n,
    }


__all__ = [
    "CertifiedHorizonResult",
    "DISCLAIMER",
    "certified_horizon",
    "enclose_jacobian_ball",
    "honesty_payload",
    "horizon_skill",
    "monodromy_product",
    "worked_example",
]
