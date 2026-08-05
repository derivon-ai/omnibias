# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""omnibias-combinatorics: exact differentiable matching / flow / matroid layers.

Assignment, transportation, min-cost-flow and matroid optimization are in **P** --
solved exactly by Hungarian / LP / max-flow / greedy (:func:`classical_optimum`). This
package does **not** claim to differentiate an NP-hard problem; the one honest limit is
narrower: the exact combinatorial argmin is piecewise-constant, so its gradient is a.e.
zero and useless for learning. The well-posed question gets a **yes-if**:

> **Yes** you can put an exact combinatorial solver inside a network and train *through*
> it **if** you relax it entropically (Sinkhorn), decode to a vertex, and report a
> certified gap -- which here is *tight* (``~0``) because these polytopes are integral.

Each layer is a three-part object:

1. a **differentiable entropic / Sinkhorn relaxation** onto the problem's polytope
   (Birkhoff / transportation / flow / matroid); as the inverse temperature
   ``beta -> inf`` the soft point collapses onto a **polytope vertex**. Unrolled for
   backprop, so a cost / weight model trains *through* it
   (:mod:`omnibias.combinatorics.jax` / :mod:`omnibias.combinatorics.torch`, bit-identical
   twins). The Sinkhorn / soft-top-k kernels are **reused** from ``omnibias-graph``.
2. a **decode to a vertex** (:func:`decode`), the *upper* bound; the exact classical
   algorithm (:func:`classical_optimum`) is the best-in-class baseline and
   :func:`brute_force_min` the small-instance vertex-enumeration cross-check.
3. a **tight LP-dual certificate** (:func:`certify_gap`): the Neumaier-Shcherbina verified
   LP dual is a rigorous *lower* bound, so ``lower <= optimum <= objective`` is a certified
   gap. It needs ``scipy`` (the exact LP solve) and the ``convex`` extra (the interval
   seal; without it the bound degrades to the valid float LP value, ``certified=False``).

Terminology: the ``beta -> inf`` hardening above is the feasibility / temperature sense
of "collapse" (a soft point becoming a 0/1 vertex), distinct from the
**founding bias collapse** (the multi-bias ``delta -> 0`` limit to the closed-form
derivative ``sigma^(K-1)``; see ``docs/theory.md`` and :mod:`omnibias.torch.unit`).

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

from omnibias.combinatorics._core.decode import (
    brute_force_min,
    classical_optimum,
    decode,
    max_flow_value,
)
from omnibias.combinatorics._core.matroids import (
    GraphicMatroid,
    Matroid,
    PartitionMatroid,
    UniformMatroid,
    independent_sets,
)
from omnibias.combinatorics._core.polytopes import PolytopeSystem
from omnibias.combinatorics.certify import certify_gap
from omnibias.combinatorics.problem import (
    AnnealSchedule,
    AssignmentProblem,
    CombinatorialCertificate,
    MatroidProblem,
    MinCostFlowProblem,
    TransportProblem,
)

try:
    __version__ = _pkg_version("omnibias-combinatorics")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "temperature collapse"

__all__ = [
    "AnnealSchedule",
    "AssignmentProblem",
    "CombinatorialCertificate",
    "GraphicMatroid",
    "Matroid",
    "MatroidProblem",
    "MinCostFlowProblem",
    "PartitionMatroid",
    "PolytopeSystem",
    "TransportProblem",
    "UniformMatroid",
    "__lineage__",
    "__version__",
    "brute_force_min",
    "certify_gap",
    "classical_optimum",
    "decode",
    "independent_sets",
    "max_flow_value",
]
