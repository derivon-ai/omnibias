# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias-graph: differentiable spectral graph operators + combinatorial relaxations.

Two families, both bit-identical across the torch and jax backends:

* **Spectral graph ops** (``ops.spectral``): the combinatorial / normalized /
  random-walk graph Laplacians, differentiable Laplacian eigenmaps
  (spectral embedding), the graph heat kernel ``exp(-t L)``, and the
  eigenvector (Rayleigh-Ritz) relaxation of ratio / normalized cut.
* **Differentiable relaxations** (``ops.relaxation``): the Sinkhorn projection
  onto doubly-stochastic matrices, Gumbel-Sinkhorn permutation sampling,
  SoftSort differentiable sorting, and a soft top-k operator -- each with a
  temperature that recovers the hard combinatorial object in the limit.

Backend ops live under ``omnibias.graph.torch`` and ``omnibias.graph.jax``.

Scope (yes-if)
--------------
These are *continuous, differentiable relaxations* and *smooth spectral*
quantities: a relaxed cut / assignment / tour value is a rigorous *lower bound* on
the discrete optimum, and certified differentiable **routing** (a relaxation + a
decoder + a reported optimality gap, ``lower <= optimum <= tour_cost``) is supported
end to end in ``omnibias-routing``. The **one honest limit** (a theorem, kept): no
poly-time differentiable map returns the *exact* NP-hard optimum -- that would imply
``P = NP``, and the exact argmin's gradient is a.e. zero, so an "exact differentiable
TSP" is ill-posed for learning. ``omnibias-graph`` itself therefore ships **only**
relaxations and exact spectral algebra -- never an exact NP-hard solver -- guarded by
an enforcement test; see ``docs/cookbook/graph-limitation.md`` and
``docs/scope-and-guarantees.md`` s6.

Relation to ``omnibias-struct`` (decided: intentionally divergent)
------------------------------------------------------------------
The soft top-k / SoftSort / Sinkhorn relaxations here and the ``logsumexp_beta``
soft-DP layers in ``omnibias-struct`` are **intentionally distinct surfaces, not a
shared substrate**. ``omnibias-graph`` relaxes *uncoupled* combinatorial objects
(sorting, assignment, top-``k``) on a flat score vector / cost matrix, with a
temperature ``tau`` in the *denominator* of the logits; ``omnibias-struct`` relaxes the
*coupled* partition function / marginals of an exponentially large structured state
space (trellis / grammar / lattice), with an inverse temperature ``beta`` *multiplying*
the scores and a closed-form ``log(N)/beta`` gap certificate. They coincide only at the
trivial plain-``softmax`` special case (a single unstructured choice), so the two are
kept as separate operators by design rather than folded onto one substrate.

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

try:
    __version__ = _pkg_version("omnibias-graph")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "temperature collapse"

__all__ = ["__lineage__", "__version__"]
