# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""omnibias-nphard: differentiable, certified heuristics for named NP-hard families.

Quadratic assignment (QAP), generalized assignment (GAP) and parallel-machine
scheduling are **NP-hard**, so no poly-time differentiable map yields the *exact*
global optimum (that would imply P = NP, and the exact argmin's gradient is a.e. zero).
Like :mod:`omnibias.qubo` / :mod:`omnibias.routing`, and unlike the P-class
:mod:`omnibias.combinatorics` (whose certificate is *tight*), this package answers the
well-posed question with a **yes, if** -- a certified gap, never an exactness claim:

1. a **differentiable annealed relaxation** -- each family reduces to a quadratic
   pseudo-Boolean (QUBO-form) energy, relaxed by :mod:`omnibias.qubo`'s soft assignment
   ``x = sigmoid(beta z) in (0, 1)^n`` descended while ``beta -> inf`` collapses it onto
   a binary vertex, *unrolled* for backprop so a model predicting the flow / cost /
   processing-times trains *through* the optimizer
   (:mod:`omnibias.nphard.jax` / :mod:`omnibias.nphard.torch`, bit-identical twins);
2. a **structure-preserving decoder** (:func:`decode`) -- Hungarian for QAP, a
   capacity-feasible rounding for GAP, argmax + LPT-repair for scheduling -- a strong
   heuristic *upper* bound, with named classical baselines (:func:`classical_optimum`)
   and an exact exponential oracle (:func:`brute_force_min`) for tiny instances;
3. a **rigorous optimality-gap certificate** (:func:`certify_gap`): a spectral / box-QP
   or Lasserre / SOS *lower* bound on the true optimum, so ``lower <= optimum <= energy``
   is a certified gap. Unlike :mod:`omnibias.combinatorics` the gap is **honestly
   non-tight** -- a weaker bound only widens it; it is never asserted zero.

A from-scratch UCT MCTS "search track" (:mod:`omnibias.nphard.search`) uses the
differentiable relaxation as an AlphaZero-style **prior** -- a heuristic search, whose
result is still handed to :func:`certify_gap` for a sound gap.

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

from omnibias.nphard._core.bound import gilmore_lawler_bound
from omnibias.nphard._core.decision import normalized_regret, spo_plus_gradient
from omnibias.nphard._core.decode import brute_force_min, classical_optimum, decode
from omnibias.nphard._core.gap import GAPProblem, gap
from omnibias.nphard._core.qap import QAPProblem, placement_qap, qap
from omnibias.nphard._core.scheduling import SchedulingProblem, schedule
from omnibias.nphard.certify import certify_gap
from omnibias.nphard.problem import AnnealSchedule, NPHardCertificate

try:
    __version__ = _pkg_version("omnibias-nphard")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "temperature collapse"

__all__ = [
    "AnnealSchedule",
    "GAPProblem",
    "NPHardCertificate",
    "QAPProblem",
    "SchedulingProblem",
    "__lineage__",
    "__version__",
    "brute_force_min",
    "certify_gap",
    "classical_optimum",
    "decode",
    "gap",
    "gilmore_lawler_bound",
    "normalized_regret",
    "placement_qap",
    "qap",
    "schedule",
    "spo_plus_gradient",
]
