# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""omnibias.geometry.atlas: a differentiable atlas of local geometries (region-wise metrics).

A bridge on :mod:`omnibias.partition`: a soft partition of unity over the coordinates plus one
per-point metric ``G_l(x)`` per region, blended into a single metric

.. math:: g(x) = \sum_l w_l(x)\, G_l(x).

Because the partition weights are non-negative and sum to one, ``g(x)`` is a **convex
combination of the region metrics** and is therefore symmetric positive-definite wherever each
``G_l(x)`` is -- a valid Riemannian metric everywhere (unit-tested). The backend builders wrap
it into the existing :class:`~omnibias.geometry.MetricSpec` / :class:`~omnibias.geometry.ManifoldSpec`,
so ``christoffel`` / ``scalar_curvature`` / ``laplace_beltrami`` / ``geodesic_rhs`` are reused
unchanged. As ``beta -> inf`` the partition hardens and the atlas approaches a piecewise metric
(a different chart per leaf).

The backend-neutral :class:`AtlasSpec` lives here; the builders live in
``omnibias.geometry.atlas.torch`` and ``omnibias.geometry.atlas.jax``
(``blended_metric`` / ``atlas_manifold``).

Terminology: the ``beta -> inf`` gate hardening is the feasibility / temperature sense of
"collapse", distinct from the **founding bias collapse** (the multi-bias ``delta -> 0`` limit
to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``). Metric derivatives are
exact forward-mode autodiff of the analytic blended metric, matching geometry today.
"""

from __future__ import annotations

from importlib.util import find_spec

if find_spec("omnibias.partition") is None:  # the optional ``atlas`` extra is not installed
    raise ImportError(
        "omnibias.geometry.atlas requires the optional 'omnibias-partition' package. "
        "Install it with:  pip install 'omnibias-geometry[atlas]'"
    )

from omnibias.geometry.atlas._core import AtlasSpec  # noqa: E402

__all__ = ["AtlasSpec"]
