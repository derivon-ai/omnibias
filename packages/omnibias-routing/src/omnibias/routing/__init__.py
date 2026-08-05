# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""omnibias-routing: certified + differentiable combinatorial routing.

The travelling-salesman tour is NP-hard, so no poly-time differentiable map yields
the *exact* optimal tour (that would imply P = NP, and the exact argmin's gradient
is a.e. zero). The sound "differentiable TSP" this package delivers is a three-part
object -- **yes, if** you accept a certified gap instead of an exactness claim:

1. a **differentiable convex relaxation** over a poly-size TSP polytope (assignment /
   single-commodity-flow / Held-Karp), solved by the omnibias temperature-collapse penalty
   *unrolled* for backprop -- so a cost model can be trained *through* the optimizer
   (:mod:`omnibias.routing.jax` / :mod:`omnibias.routing.torch`, bit-identical twins);
2. a **heuristic decoder** -- nearest-neighbour + 2-opt / or-opt -- that rounds the
   fractional arc-use to a valid tour (:func:`decode_tour`), an *upper* bound;
3. a **rigorous optimality-gap certificate** (:func:`certify_tour_gap`): the
   Neumaier-Shcherbina verified LP dual bound is a *lower* bound on the true optimum,
   so ``lower <= optimum <= tour_cost`` is a certified gap -- never asserted zero, and
   honest about relaxation strength (a weaker relaxation only widens the gap).

For small ``n`` the exact optimum is available (:func:`held_karp_dp`) to self-check
the sandwich and to score decision-focused :func:`normalized_regret`. The relaxation
layers need a ``jax`` / ``torch`` backend; the certificate needs ``scipy`` (the exact
LP solve) and the ``convex`` extra (the rigorous interval seal; without it the bound
degrades gracefully to the valid float LP value with ``certified=False``).

Terminology: "temperature-collapse penalty" above is the feasibility sense of
"collapse" (a hard-hinge / ``beta -> inf`` constraint force), distinct from the
**founding bias collapse** (the multi-bias ``delta -> 0`` limit to
``sigma^(K-1)``, a derivative; see ``docs/theory.md``).

.. important::

    **Bit-parity with the PyTorch twin requires 64-bit JAX** --
    ``jax.config.update("jax_enable_x64", True)`` before the first JAX array is
    created (or ``JAX_ENABLE_X64=1``). JAX otherwise truncates to ``float32``
    while PyTorch uses ``float64``, so the twins stay internally consistent but
    agree only to ``float32`` tolerance. Where a value feeds a threshold, a
    rounding step or an ``argmax``, that is enough to change the decision rather
    than just the last digits. See :mod:`omnibias.jax.precision`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

from omnibias.routing._core.decision import (
    edge_matrix,
    normalized_regret,
    optimal_tour_costs,
    spo_plus_gradient,
)
from omnibias.routing._core.decode import (
    decode_tour,
    held_karp_dp,
    is_valid_tour,
    nearest_neighbor,
    tour_cost,
    two_opt,
)
from omnibias.routing._core.relax_systems import RelaxSystem, build_system
from omnibias.routing.certify import certify_tour_gap
from omnibias.routing.problem import (
    HeldKarpCertificate,
    RelaxationSchedule,
    RoutingProblem,
    TourSolution,
)

try:
    __version__ = _pkg_version("omnibias-routing")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "temperature collapse"

__all__ = [
    "HeldKarpCertificate",
    "RelaxSystem",
    "RelaxationSchedule",
    "RoutingProblem",
    "TourSolution",
    "__lineage__",
    "__version__",
    "build_system",
    "certify_tour_gap",
    "decode_tour",
    "edge_matrix",
    "held_karp_dp",
    "is_valid_tour",
    "nearest_neighbor",
    "normalized_regret",
    "optimal_tour_costs",
    "spo_plus_gradient",
    "tour_cost",
    "two_opt",
]
