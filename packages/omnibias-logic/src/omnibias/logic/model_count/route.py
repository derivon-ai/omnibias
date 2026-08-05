# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""A **sound-only** auto-router over the (weighted) #SAT counting methods.

:func:`count` dispatches a :class:`~omnibias.logic.model_count.problem.ModelCountProblem` to
the cheapest method that is still **worst-case sound**, and always returns a
:class:`CountResult` tagged with which guarantee it earned:

* ``guarantee="exact"`` -- an exact ``int`` / :class:`~fractions.Fraction` count from a
  tractable regime (affine/XOR, bounded-treewidth DP) or a completed DPLL search;
* ``guarantee="certified_enclosure"`` -- a rigorous ``lower <= #models <= upper`` sandwich
  from :func:`~omnibias.logic.count_enclosure` when no exact method finishes in budget.

Auto policy (``mode="auto"``): unweighted **affine/XOR** system -> the GF(2) fast path;
else the **bounded-treewidth DP** if its heuristic width fits ``max_width``; else the
**DPLL** counter under ``node_budget``; else the **certified enclosure** fallback. Every
branch is sound, so a ``CountResult`` can be trusted regardless of which method won.

The statistical estimators in :mod:`omnibias.logic.approx` are **never** reached from here --
they return a different, explicitly non-sound type and must be called directly.

Warm start: with ``warm_start=True`` the router derives a DPLL branching order from the
annealed :func:`~omnibias.logic.torch.sat_relaxation` (or its jax twin) -- most-polarised
variable first. This affects search *speed only*; the exact count is invariant to the order,
and absent a tensor backend the router silently uses the default order.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from omnibias.logic.model_count.enclosure import count_enclosure
from omnibias.logic.model_count.exact import CountBudgetExceeded, count_models_exact
from omnibias.logic.model_count.treewidth import TreewidthTooLarge, treewidth_model_count
from omnibias.logic.model_count.xor import detect_xor_system, xor_model_count

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.logic.model_count.problem import ModelCountProblem

_TINY = 1e-9

_MODES = ("auto", "xor", "treewidth", "dpll", "enclosure")


@dataclass(frozen=True)
class CountResult:
    r"""A tagged, worst-case-**sound** model-count result.

    Attributes
    ----------
    method:
        The method that produced the result (``"affine_gf2"``, ``"treewidth_dp"``,
        ``"dpll"``, or ``"inclusion_exclusion"`` / ``"trivial"``).
    guarantee:
        ``"exact"`` (``value`` is the exact count) or ``"certified_enclosure"`` (only the
        rigorous ``lower`` / ``upper`` bracket is known). Both are worst-case sound.
    value:
        The exact ``int`` / :class:`~fractions.Fraction` count when ``guarantee == "exact"``,
        else ``None``.
    lower, upper:
        A rigorous bracket on the count as floats. When ``guarantee == "exact"`` they are the
        (float view of the) exact value.
    """

    method: str
    guarantee: str
    value: int | Fraction | None
    lower: float
    upper: float

    @property
    def is_exact(self) -> bool:
        """Whether this particular result is an exact count (not merely an enclosure)."""
        return self.guarantee == "exact"

    @property
    def is_sound(self) -> bool:
        """Always ``True``: every router result is worst-case sound (exact or enclosure)."""
        return self.guarantee in ("exact", "certified_enclosure")

    @property
    def width(self) -> float:
        """The bracket width ``upper - lower`` (``0`` for an exact result)."""
        return self.upper - self.lower

    def contains(self, count: float) -> bool:
        """Whether a candidate ``count`` lies inside the sound bracket."""
        return self.lower - _TINY <= count <= self.upper + _TINY


def _exact_result(method: str, value: int | Fraction) -> CountResult:
    as_float = float(value)
    return CountResult(
        method=method, guarantee="exact", value=value, lower=as_float, upper=as_float
    )


def _relaxation_branch_order(problem: ModelCountProblem) -> list[int] | None:
    """A most-polarised-first branch order from the annealed relaxation, or ``None``.

    Backend-guarded: tries the torch relaxation, then the jax twin; if neither backend is
    importable (or anything goes wrong) returns ``None`` so the caller uses the default order.
    """
    soft: np.ndarray | None = None
    try:
        from omnibias.logic.torch import sat_relaxation as torch_relaxation

        soft = np.asarray(torch_relaxation(problem).detach().cpu().numpy(), dtype=float)
    except Exception:  # noqa: BLE001 - any backend failure -> default order
        try:
            from omnibias.logic.jax import sat_relaxation as jax_relaxation

            soft = np.asarray(jax_relaxation(problem), dtype=float)
        except Exception:  # noqa: BLE001
            return None
    if soft is None:
        return None
    soft = soft.reshape(-1)
    if soft.shape[0] != problem.n:
        return None
    polarisation = np.abs(soft - 0.5)
    return [int(i) + 1 for i in np.argsort(-polarisation)]


def count(
    problem: ModelCountProblem,
    *,
    mode: str = "auto",
    max_width: int = 18,
    node_budget: int | None = 200_000,
    order: int = 2,
    warm_start: bool = False,
) -> CountResult:
    r"""Route ``problem`` to the cheapest sound counter and return a tagged :class:`CountResult`.

    Parameters
    ----------
    problem:
        The instance to count.
    mode:
        ``"auto"`` (default) or one of the specific methods ``"xor"`` / ``"treewidth"`` /
        ``"dpll"`` / ``"enclosure"``. A specific mode that does not apply raises.
    max_width:
        Treewidth cap for the exact DP path.
    node_budget:
        Branch-node cap for the DPLL path (``None`` = unbounded); exceeding it falls back to
        the enclosure under ``mode="auto"``.
    order:
        Inclusion-exclusion truncation order for the enclosure fallback.
    warm_start:
        Derive the DPLL branch order from the annealed relaxation (backend-guarded).
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")

    branch_order = _relaxation_branch_order(problem) if warm_start else None

    if mode in ("auto", "xor") and not problem.is_weighted:
        xor_clauses = detect_xor_system(problem)
        if xor_clauses is not None:
            return _exact_result("affine_gf2", xor_model_count(xor_clauses, problem.n))
        if mode == "xor":
            raise ValueError(
                "mode='xor' requires an unweighted affine (XOR) system; this CNF is not one "
                "(use mode='auto' to fall back to a CNF-native counter)."
            )
    elif mode == "xor":
        raise ValueError("mode='xor' does not support weighted counting (affine count is unweighted)")

    if mode in ("auto", "treewidth"):
        try:
            value, _width = treewidth_model_count(problem, max_width=max_width)
            return _exact_result("treewidth_dp", value)
        except TreewidthTooLarge:
            if mode == "treewidth":
                raise

    if mode in ("auto", "dpll"):
        try:
            value = count_models_exact(problem, branch_order=branch_order, node_budget=node_budget)
            return _exact_result("dpll", value)
        except CountBudgetExceeded:
            if mode == "dpll":
                raise

    certificate = count_enclosure(problem, order=order)
    return CountResult(
        method=certificate.method,
        guarantee="certified_enclosure",
        value=None,
        lower=certificate.lower,
        upper=certificate.upper,
    )


__all__ = ["CountResult", "count"]
