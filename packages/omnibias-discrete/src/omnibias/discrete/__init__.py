# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""omnibias-discrete: the shared differentiable + certified discrete-optimization substrate.

Minimizing a pseudo-Boolean energy ``E(x)`` over ``x in {0, 1}^n`` is NP-hard, so no
poly-time differentiable map yields the *exact* global optimum (that would imply
P = NP, and the exact argmin's gradient is a.e. zero). The sound object this substrate
delivers is the ``encode -> relax -> decode -> certify`` pipeline -- **yes, if** you
accept a certified gap instead of an exactness claim:

1. a **differentiable annealed relaxation** (:mod:`omnibias.discrete.jax` /
   :mod:`omnibias.discrete.torch`, bit-identical twins): given a closed-form energy
   gradient, :func:`anneal_descent` descends ``x = sigmoid(beta theta)`` while
   ``beta -> inf`` collapses it onto a binary vertex, *unrolled* for backprop;
2. a **heuristic decoder** -- rounding + 1-flip local search (:func:`decode`), an
   *upper* bound (:func:`brute_force_min` is the exact small-``n`` oracle);
3. a **rigorous optimality-gap certificate** (:func:`certify_gap`): a Lasserre / SOS
   bound over the Boolean hypercube (:mod:`omnibias.sos`), seeded by the always-valid
   :func:`negative_coeff_lower_bound`, is a *lower* bound on the true optimum, so
   ``lower <= optimum <= energy`` is a certified gap -- never asserted zero.

Anything implementing the :class:`DiscreteProblem` seam (``n`` + ``energy`` +
``to_polynomial``) plugs into the whole pipeline; ``omnibias-qubo`` and the in-tree
:mod:`omnibias.discrete.maxsat` front-end are the first two consumers.

Backend-neutral helpers shared by several consumers also live here so they are written
once: the decision-focused (predict-then-optimize) :func:`spo_plus_subgradient` /
:func:`mean_normalized_regret` (bound to each consumer's exact oracle in ``omnibias-nphard`` /
``omnibias-routing``); the :class:`UnionFind` / :func:`is_forest` graph primitive; and the
representation-neutral matroid independence / rank kernel (:mod:`omnibias.discrete.matroid`)
that is the single canonical definition of the uniform / partition / graphic families behind
*both* the polytope lens (``omnibias-combinatorics``) and the greedy-oracle lens
(``omnibias-submodular``).

Terminology: the relaxation's ``sigmoid(beta z)``, ``beta -> inf`` is the feasibility /
temperature sense of "collapse" (a soft indicator hardening to a 0/1 step), distinct
from the **founding bias collapse** (the multi-bias ``delta -> 0`` limit of an ``OMBU``
to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).

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

from omnibias.discrete._core.bound import (
    gershgorin_min_eig_lower,
    lasserre_lower_bound,
    negative_coeff_lower_bound,
)
from omnibias.discrete._core.decision import mean_normalized_regret, spo_plus_subgradient
from omnibias.discrete._core.decode import (
    brute_force_min,
    decode,
    energy,
    flip_deltas,
    is_binary,
    one_flip_descent,
    round_relaxed,
)
from omnibias.discrete._core.problem import DiscreteProblem, boolean_constraints
from omnibias.discrete._core.relax import INIT_THETA_SCALE, initial_theta
from omnibias.discrete._core.schedule import AnnealSchedule
from omnibias.discrete._core.solution import DiscreteSolution, GapCertificate
from omnibias.discrete._core.union_find import UnionFind, is_forest
from omnibias.discrete.certify import certify_gap

try:
    __version__ = _pkg_version("omnibias-discrete")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "temperature collapse"

__all__ = [
    "AnnealSchedule",
    "DiscreteProblem",
    "DiscreteSolution",
    "GapCertificate",
    "INIT_THETA_SCALE",
    "UnionFind",
    "__lineage__",
    "__version__",
    "boolean_constraints",
    "brute_force_min",
    "certify_gap",
    "decode",
    "energy",
    "flip_deltas",
    "gershgorin_min_eig_lower",
    "initial_theta",
    "is_binary",
    "is_forest",
    "lasserre_lower_bound",
    "mean_normalized_regret",
    "negative_coeff_lower_bound",
    "one_flip_descent",
    "round_relaxed",
    "spo_plus_subgradient",
]
