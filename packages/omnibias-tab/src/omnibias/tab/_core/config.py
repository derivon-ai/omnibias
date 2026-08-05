# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-agnostic configuration for a soft decision-tree ensemble.

:class:`SoftTreeConfig` is the single shape descriptor shared by the numpy reference
forward, the torch trainable module, and the jax functional twin, so all three present
an identical surface. It only holds plain data.

Terminology: the ensemble's split gate ``sigmoid(beta * (w.x - t))`` hardens as
``beta -> inf`` -- the feasibility / temperature sense of "collapse" (a soft indicator
becoming a 0/1 step), distinct from the **founding bias collapse** (the multi-bias
``delta -> 0`` limit of an ``OMBU`` to the closed-form derivative ``sigma^(K-1)``; see
``docs/theory.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

_TASKS = ("binary", "multiclass", "regression")


@dataclass(frozen=True)
class SoftTreeConfig:
    r"""Shape + task descriptor for an ensemble of oblivious soft decision trees.

    The model is an ensemble of ``n_trees`` oblivious (shared-per-level) soft trees, each
    of ``depth`` oblique split gates. ``depth == 1`` is the **additive** tier -- a pure
    sum-of-sigmoids (Linear -> Sigmoid -> Linear), directly certifiable; ``depth >= 2`` is
    the **multiplicative** tier, whose ``2**depth`` leaf memberships are products of gates
    (native feature interactions).

    Attributes
    ----------
    n_features:
        Input dimension ``d``.
    n_trees:
        Number of soft trees summed by the ensemble.
    depth:
        Gates per tree. ``1`` -> additive; ``>= 2`` -> multiplicative (``2**depth`` leaves).
    task:
        ``"binary"`` (scalar logit), ``"multiclass"`` (``n_outputs`` softmax logits) or
        ``"regression"`` (``n_outputs`` real outputs).
    n_outputs:
        Output width ``k``. Forced to ``1`` for ``"binary"``; ``>= 2`` for ``"multiclass"``.
    beta_init, beta_final:
        Inverse-temperature (gate sharpness) at the start / end of the anneal. The
        ``beta -> inf`` limit is the feasibility-sense collapse to hard splits.
    anneal_steps:
        Training steps over which ``beta`` is ramped ``beta_init -> beta_final`` (geometric).
    leaf_l2:
        L2 shrinkage on leaf values (a light, convexifying regulariser).
    seed:
        RNG seed for parameter initialisation.
    """

    n_features: int
    n_trees: int = 16
    depth: int = 1
    task: str = "binary"
    n_outputs: int = 1
    beta_init: float = 1.0
    beta_final: float = 32.0
    anneal_steps: int = 50
    leaf_l2: float = 1e-4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {self.n_features}")
        if self.n_trees < 1:
            raise ValueError(f"n_trees must be >= 1, got {self.n_trees}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if self.task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}, got {self.task!r}")
        if self.task == "binary" and self.n_outputs != 1:
            raise ValueError("binary task requires n_outputs == 1 (a single logit)")
        if self.task == "multiclass" and self.n_outputs < 2:
            raise ValueError("multiclass task requires n_outputs >= 2")
        if self.n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1, got {self.n_outputs}")
        if not (self.beta_init > 0.0 and self.beta_final > 0.0):
            raise ValueError("beta_init and beta_final must be positive")
        if self.anneal_steps < 1:
            raise ValueError("anneal_steps must be >= 1")

    @property
    def n_leaves(self) -> int:
        """Leaves per tree, ``2**depth``."""
        return 1 << self.depth

    @property
    def is_additive(self) -> bool:
        """``True`` for the certifiable depth-1 sum-of-sigmoids tier."""
        return self.depth == 1

    def beta_at(self, step: int) -> float:
        """Geometric ``beta`` ramp: ``beta_init`` at step 0 -> ``beta_final`` at the end."""
        if self.anneal_steps <= 1:
            return float(self.beta_final)
        frac = min(max(step, 0), self.anneal_steps - 1) / (self.anneal_steps - 1)
        return float(self.beta_init * (self.beta_final / self.beta_init) ** frac)


__all__ = ["SoftTreeConfig"]
