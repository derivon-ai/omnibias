# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Backend-agnostic :class:`AtlasSpec` -- a soft partition + one metric per region.

An :class:`AtlasSpec` is a differentiable *atlas of local geometries*: a soft partition of
unity over the coordinates (from :mod:`omnibias.partition`) plus one per-point metric
callable ``G_l(x)`` per region. The blended metric

.. math:: g(x) = \sum_l w_l(x)\, G_l(x)

is a **convex combination of the region metrics** (weights ``w_l(x) >= 0`` sum to one), so if
every ``G_l(x)`` is symmetric positive-definite then ``g(x)`` is too -- the blend is a valid
Riemannian metric everywhere (stated and unit-tested). The backend builders
(:mod:`omnibias.geometry.atlas.torch` / ``.jax``) wrap it into the existing
:class:`~omnibias.geometry.MetricSpec` / :class:`~omnibias.geometry.ManifoldSpec`, so every
downstream operator (``christoffel`` / ``scalar_curvature`` / ``laplace_beltrami`` /
``geodesic_rhs``) is reused unchanged.

Honesty note (matching geometry today): metric derivatives are exact **forward-mode
autodiff** of the analytic blended metric, not a sigma-tower closed form. The ``beta -> inf``
gate hardening is the feasibility / temperature sense of "collapse", distinct from the
**founding bias collapse** (the multi-bias ``delta -> 0`` limit to the closed-form derivative
``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.geometry._core.manifold import MetricFn
from omnibias.partition._core.params import PartitionParams


@dataclass(frozen=True)
class AtlasSpec:
    r"""A soft partition + one per-point metric callable per region (a differentiable atlas).

    Parameters
    ----------
    partition:
        The :class:`~omnibias.partition.PartitionParams` whose ``2**depth`` regions index the
        atlas charts.
    region_metrics:
        One per-point metric callable ``G_l(x)`` per region (``x`` of shape ``(d,)`` ->
        ``(d, d)``), written with backend ops so it is ``vmap`` / autodiff compatible. Each
        should return a symmetric positive-definite matrix for the blend to be a metric.
    beta:
        Gate sharpness for the partition weights (defaults to the partition config's
        ``beta_final``). Larger -> sharper region boundaries.
    name:
        Human-readable label.
    """

    partition: PartitionParams
    region_metrics: Sequence[MetricFn]
    beta: float | None = None
    name: str = "atlas"

    def __post_init__(self) -> None:
        if len(self.region_metrics) != self.partition.n_regions:
            raise ValueError(
                f"expected {self.partition.n_regions} region metrics (2**depth), "
                f"got {len(self.region_metrics)}"
            )

    @property
    def dim(self) -> int:
        return int(self.partition.n_features)

    @property
    def n_regions(self) -> int:
        return int(self.partition.n_regions)

    def beta_value(self) -> float:
        return float(self.partition.config.beta_final if self.beta is None else self.beta)


__all__ = ["AtlasSpec"]
