# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Numerical design objectives for fixed additive information blocks.

For eligible cardinality log-det selection, the copyleft submodular consumer
provides the optimization adapter. No approximation factor is attached here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def information_matrix(blocks: Array, weights: Array, prior: Array) -> Array:
    a = np.asarray(blocks, dtype=float)
    w = np.asarray(weights, dtype=float)
    p = np.asarray(prior, dtype=float)
    if (
        a.ndim != 3
        or a.shape[1] != a.shape[2]
        or w.shape != (a.shape[0],)
        or p.shape != a.shape[1:]
    ):
        raise ValueError("expected blocks (c,p,p), weights (c,), prior (p,p)")
    if not all(np.all(np.isfinite(v)) for v in (a, w, p)) or np.any(w < 0):
        raise ValueError("finite arrays and nonnegative weights required")
    if not np.allclose(a, a.transpose(0, 2, 1), atol=1e-12, rtol=0) or not np.allclose(
        p, p.T, atol=1e-12, rtol=0
    ):
        raise ValueError("symmetric information required")
    if np.min(np.linalg.eigvalsh(a)) < -1e-12:
        raise ValueError("blocks must be positive semidefinite")
    with np.errstate(over="ignore", invalid="ignore"):
        out = p + np.einsum("c,cij->ij", w, a)
    if not np.all(np.isfinite(out)):
        raise ValueError("finite total information required; rescale the information")
    try:
        np.linalg.cholesky(out)
    except np.linalg.LinAlgError as exc:
        raise ValueError("total information must be positive definite") from exc
    return np.asarray(out, dtype=float)


def design_score(blocks: Array, weights: Array, prior: Array, *, criterion: str = "D") -> float:
    """Higher is better: logdet, negative trace inverse, or smallest eigenvalue."""
    mat = information_matrix(blocks, weights, prior)
    with np.errstate(over="ignore", invalid="ignore"):
        if criterion == "D":
            score = float(np.linalg.slogdet(mat)[1])
        elif criterion == "A":
            score = -float(np.trace(np.linalg.solve(mat, np.eye(mat.shape[0]))))
        elif criterion == "E":
            score = float(np.linalg.eigvalsh(mat)[0])
        else:
            raise ValueError("criterion must be D, A or E")
    if not np.isfinite(score):
        raise ValueError("finite design scores required; rescale the information")
    return score


@dataclass(frozen=True)
class DesignSelection:
    """Finite numerical selection, with algorithm and enumeration scope explicit."""

    indices: tuple[int, ...]
    weights: Array | None
    score: float | None
    criterion: str
    method: str
    budget: int
    status: Literal["selected", "budget_exceeded"]
    enumeration_complete: bool
    evaluated_subsets: int
    required_subsets: int
    certified: bool = False


def select_design(
    blocks: Array,
    prior: Array,
    budget: int,
    *,
    criterion: str = "D",
    method: str = "greedy",
    max_subsets: int = 100_000,
) -> DesignSelection:
    """Select exactly ``budget`` independent information blocks using D/A/E.

    The prior must be positive definite, so every intermediate design has a
    valid objective. Greedy evaluates all one-block additions at each step.
    Exhaustive evaluates every subset of the requested cardinality. Numerical
    ties choose the lexicographically first subset (smallest next index for
    greedy). No approximation factor or rigorous optimality proof is attached.

    ``max_subsets`` caps candidate-subset evaluations. Work exceeding that cap
    is refused before selection, with no partial design. Initial input/prior
    validation is outside this counter. An empty design requires one reported
    evaluation and returns the prior's score. ``enumeration_complete`` is true
    only after exhaustive enumeration, even when a greedy run fills its budget.
    """
    a = np.asarray(blocks, dtype=float)
    p = np.asarray(prior, dtype=float)
    if a.ndim != 3 or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError("nonempty information blocks (c,p,p) required")
    count = a.shape[0]
    if (
        type(budget) is not int
        or budget < 0
        or budget > count
        or type(max_subsets) is not int
        or max_subsets < 0
    ):
        raise ValueError("integer budget in [0,c] and nonnegative max_subsets required")
    if method not in ("greedy", "exhaustive"):
        raise ValueError("method must be greedy or exhaustive")
    empty = np.zeros(count)
    baseline = design_score(a, empty, p, criterion=criterion)
    if not np.isfinite(baseline):
        raise ValueError("finite design scores required; rescale the information")
    required = (
        comb(count, budget)
        if method == "exhaustive"
        else max(1, budget * count - budget * (budget - 1) // 2)
    )
    if required > max_subsets:
        return DesignSelection(
            (), None, None, criterion, method, budget, "budget_exceeded", False, 0, required
        )

    def score_subset(indices: tuple[int, ...]) -> tuple[float, Array]:
        weights = np.zeros(count)
        weights[list(indices)] = 1.0
        value = design_score(a, weights, p, criterion=criterion)
        if not np.isfinite(value):
            raise ValueError("finite design scores required; rescale the information")
        return value, weights

    selected: tuple[int, ...] = ()
    weights = empty
    score = baseline
    evaluated = 0
    if budget == 0:
        evaluated = 1
    elif method == "exhaustive":
        score = -float("inf")
        for indices in combinations(range(count), budget):
            value, candidate_weights = score_subset(indices)
            evaluated += 1
            if value > score:
                selected, score, weights = indices, value, candidate_weights
    else:
        for _ in range(budget):
            best_score = -float("inf")
            best = selected
            best_weights = weights
            for index in range(count):
                if index in selected:
                    continue
                candidate = tuple(sorted((*selected, index)))
                value, candidate_weights = score_subset(candidate)
                evaluated += 1
                if value > best_score:
                    best, best_score, best_weights = candidate, value, candidate_weights
            selected, score, weights = best, best_score, best_weights
    return DesignSelection(
        selected,
        weights,
        score,
        criterion,
        method,
        budget,
        "selected",
        method == "exhaustive",
        evaluated,
        required,
    )


def design_gradient(blocks: Array, weights: Array, prior: Array, *, criterion: str = "D") -> Array:
    mat = information_matrix(blocks, weights, prior)
    if criterion == "E":
        eigenvalues, eigenvectors = np.linalg.eigh(mat)
        if len(eigenvalues) > 1 and eigenvalues[1] - eigenvalues[0] <= (
            64 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(mat, 2)))
        ):
            raise ValueError("E gradient requires a simple smallest eigenvalue")
        direction = eigenvectors[:, 0]
        return np.asarray(np.einsum("i,cij,j->c", direction, blocks, direction), dtype=float)
    inv = np.linalg.solve(mat, np.eye(mat.shape[0]))
    if criterion == "D":
        return np.einsum("ij,cji->c", inv, blocks)  # type: ignore[no-any-return]
    if criterion == "A":
        return np.einsum("ij,cjk,ki->c", inv, blocks, inv)  # type: ignore[no-any-return]
    raise ValueError("criterion must be D, A or E")


def differentiable_design_score(
    blocks: Any, weights: Any, prior: Any, *, backend: str, criterion: str = "D"
) -> Any:
    """Backend objective on already validated PSD blocks/positive total information.

    This numerical linear algebra is differentiable in weights; call the numpy
    validator at ingestion. Runtime tensor values may be traced. D/A are smooth
    on positive definite matrices. E uses the smallest eigenvalue and is
    differentiable where it is simple; at multiplicity it is nonsmooth and the
    backend's selected derivative is not a unique gradient. ``design_gradient``
    explicitly refuses that ambiguous E case.
    """
    if backend == "torch":
        import torch

        mat = prior + torch.einsum("c,cij->ij", weights, blocks)
        if criterion == "D":
            return torch.linalg.slogdet(mat)[1]
        if criterion == "A":
            return -torch.trace(torch.linalg.inv(mat))
        if criterion == "E":
            return torch.linalg.eigvalsh(mat)[0]
    elif backend == "jax":
        import jax.numpy as jnp

        mat = prior + jnp.einsum("c,cij->ij", weights, blocks)
        if criterion == "D":
            return jnp.linalg.slogdet(mat)[1]
        if criterion == "A":
            return -jnp.trace(jnp.linalg.inv(mat))
        if criterion == "E":
            return jnp.linalg.eigvalsh(mat)[0]
    raise ValueError("backend must be torch/jax and criterion D/A/E")


__all__ = [
    "DesignSelection",
    "design_gradient",
    "design_score",
    "differentiable_design_score",
    "information_matrix",
    "select_design",
]
