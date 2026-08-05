# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Backend-agnostic configuration for a soft partition of unity.

:class:`PartitionConfig` is the single shape descriptor shared by the numpy reference
weights, the torch twin, and the jax twin, so all three present an identical surface. It
holds only plain data.

``split_kind`` selects the geometry of the ``depth`` split gates:

* ``"oblique"`` -- dense direction ``w`` per gate (a general hyperplane ``w.x = t``);
* ``"axis"`` -- each gate reads a single feature (``x[f] > t``), the interpretable /
  heterogeneous-robust mode with clean ``if x[f] > t`` hardened rules;
* ``"sparse"`` -- oblique storage that a trainer sparsifies with an L1 penalty (partition
  itself does not train; the flag advertises intent and drives the rule export).

Terminology: the split gate ``sigmoid(beta * (w.x - t))`` hardens as ``beta -> inf`` -- the
feasibility / temperature sense of "collapse" (a soft indicator becoming a 0/1 step),
distinct from the **founding bias collapse** (the multi-bias ``delta -> 0`` limit of an
``OMBU`` to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

_SPLIT_KINDS = ("oblique", "axis", "sparse")


@dataclass(frozen=True)
class PartitionConfig:
    r"""Shape + split-geometry descriptor for a soft partition of unity.

    A single soft tree of ``depth`` oblique gates routes ``R^{n_features}`` into
    ``2**depth`` regions; the region weights are a partition of unity (non-negative, sum to
    one) that hardens to a crisp partition as ``beta -> inf``.

    Attributes
    ----------
    n_features:
        Input dimension ``d``.
    depth:
        Number of split gates. The partition has ``2**depth`` regions.
    split_kind:
        ``"oblique"`` (dense hyperplanes), ``"axis"`` (single-feature thresholds) or
        ``"sparse"`` (oblique storage, L1-sparsified by a trainer).
    beta_init, beta_final:
        Gate sharpness (inverse temperature) at the start / end of the anneal. The
        ``beta -> inf`` limit is the feasibility-sense collapse to a hard partition.
    anneal_steps:
        Steps over which ``beta`` ramps ``beta_init -> beta_final`` (geometric).
    seed:
        RNG seed for parameter initialisation.
    """

    n_features: int
    depth: int = 1
    split_kind: str = "oblique"
    beta_init: float = 1.0
    beta_final: float = 32.0
    anneal_steps: int = 50
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {self.n_features}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if self.split_kind not in _SPLIT_KINDS:
            raise ValueError(f"split_kind must be one of {_SPLIT_KINDS}, got {self.split_kind!r}")
        if not (self.beta_init > 0.0 and self.beta_final > 0.0):
            raise ValueError("beta_init and beta_final must be positive")
        if self.anneal_steps < 1:
            raise ValueError("anneal_steps must be >= 1")

    @property
    def n_regions(self) -> int:
        r"""Number of regions, ``2**depth``."""
        return 1 << self.depth

    @property
    def is_axis(self) -> bool:
        r"""``True`` for the single-feature (axis-aligned) split mode."""
        return self.split_kind == "axis"

    def beta_at(self, step: int) -> float:
        r"""Geometric ``beta`` ramp: ``beta_init`` at step 0 -> ``beta_final`` at the end."""
        if self.anneal_steps <= 1:
            return float(self.beta_final)
        frac = min(max(step, 0), self.anneal_steps - 1) / (self.anneal_steps - 1)
        return float(self.beta_init * (self.beta_final / self.beta_init) ** frac)


__all__ = ["PartitionConfig"]
