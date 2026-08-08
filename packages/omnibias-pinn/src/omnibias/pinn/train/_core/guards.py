# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Training guards that catch unphysical PINN collapse modes (pure numpy).

A PINN can converge to a trivial constant (or near-constant) solution that
satisfies soft residual / BC penalties without resolving the dynamics. These
guards *report* that failure mode; they never raise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrivialSolutionVerdict:
    """Outcome of :func:`trivial_solution_guard`.

    Attributes
    ----------
    is_trivial
        ``True`` when the solution energy / variance has collapsed relative to
        the initial-condition reference below ``ratio_threshold``.
    solution_energy
        Mean square of the solution samples (or variance when ``mode="variance"``).
    reference_energy
        Mean square (or variance) of the IC / reference samples.
    ratio
        ``solution_energy / max(reference_energy, eps)``.
    mode
        Which statistic was compared (``"energy"`` or ``"variance"``).
    """

    is_trivial: bool
    solution_energy: float
    reference_energy: float
    ratio: float
    mode: str


def trivial_solution_guard(
    solution: np.ndarray,
    reference: np.ndarray,
    *,
    ratio_threshold: float = 1e-3,
    mode: str = "energy",
    eps: float = 1e-30,
) -> TrivialSolutionVerdict:
    """Compare solution energy / variance against an IC (or reference) sample.

    Parameters
    ----------
    solution
        Flattened or multi-D solution samples.
    reference
        Initial-condition (or other reference) samples of comparable scale.
    ratio_threshold
        Flag trivial when ``solution_stat / reference_stat < ratio_threshold``.
    mode
        ``"energy"`` uses mean-square; ``"variance"`` uses sample variance
        (catches collapse to a non-zero constant).
    eps
        Floor on the reference statistic to avoid division by zero.
    """
    if mode not in ("energy", "variance"):
        raise ValueError(f"mode must be 'energy' or 'variance', got {mode!r}")
    if ratio_threshold <= 0.0:
        raise ValueError(f"ratio_threshold must be > 0, got {ratio_threshold}")
    sol = np.asarray(solution, dtype=float).reshape(-1)
    ref = np.asarray(reference, dtype=float).reshape(-1)
    if sol.size == 0 or ref.size == 0:
        raise ValueError("solution and reference must be non-empty")
    if mode == "energy":
        sol_stat = float(np.mean(sol * sol))
        ref_stat = float(np.mean(ref * ref))
    else:
        sol_stat = float(np.var(sol))
        ref_stat = float(np.var(ref))
    ratio = sol_stat / max(ref_stat, float(eps))
    return TrivialSolutionVerdict(
        is_trivial=bool(ratio < ratio_threshold),
        solution_energy=sol_stat,
        reference_energy=ref_stat,
        ratio=ratio,
        mode=mode,
    )


__all__ = [
    "TrivialSolutionVerdict",
    "trivial_solution_guard",
]
