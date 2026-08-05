# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""omnibias-logic: differentiable + certified Boolean logic.

Two classic hard problems on a CNF, each answered as a **yes-if** -- a certified object, not
an exactness claim:

1. **weighted MaxSAT**, *re-exported unchanged* from :mod:`omnibias.discrete.maxsat` (the
   weighted-violation ``DiscreteProblem`` on the ``omnibias-discrete`` substrate): build an
   instance with :func:`max_sat`, relax it (the annealed ``sigmoid(beta z)``, ``beta -> inf``
   penalty; torch + jax twins), :func:`decode` a min-violation assignment (upper bound), and
   :func:`certify_gap` the Lasserre / SOS **optimality gap** ``lower <= optimum <= energy``.
   Minimizing the violation is NP-hard, so this is never an exact-optimum (``P = NP``) claim.

2. **(weighted) #SAT / model counting**, *added here*: :func:`model_count` builds a
   :class:`ModelCountProblem`, :func:`count_enclosure` returns a rigorous
   ``lower <= #models <= upper`` inclusion-exclusion (Bonferroni) enclosure
   (:class:`CountCertificate`), and :func:`exact_model_count` is the exact ``O(2^n)`` oracle
   that self-checks it. Exact (weighted) model counting is ``#P``-hard, so the deliverable is
   a certified **enclosure**, never a poly-time exact-count claim; a lower order only *widens*
   the enclosure, never fakes it.

   For the tractable *fragments*, exact counters are also exported: :func:`xor_model_count`
   (affine / XOR fragment via GF(2) rank -- unweighted), :func:`treewidth_model_count`
   (bounded-treewidth DP -- weighted), and :func:`count_models_exact` (component-caching
   #DPLL -- weighted, exponential worst case). :func:`count` is a **sound-only** router that
   auto-picks the cheapest of these (falling back to the enclosure) and returns a tagged
   :class:`CountResult`. Statistical, **NOT worst-case sound** estimators are quarantined in
   :mod:`omnibias.logic.approx` and are *not* re-exported here -- import them explicitly.

   A :class:`CountCertificate` can be :func:`sealed <seal_count_certificate>` into the
   tamper-evident v1 certificate format and handed to the Mathlib-free Lean kernel via
   :func:`check_certificate`: a tight, unweighted enclosure yields a kernel-checked exact-count
   *integer identity* ``Z0 - S_1 + S_2 - ... = #models`` (finite inclusion-exclusion; includes
   certified UNSAT), otherwise a positive lower bound yields a kernel-checked satisfiability
   sign. Use :func:`verify_certificate_digest` to detect tampering; ``theorem_prover_verified``
   is earned only by a genuine ``lake build`` pass and degrades gracefully with no toolchain.

Both differentiable relaxations reuse the substrate's ``anneal_descent``; their
``sigmoid(beta z)``, ``beta -> inf`` is the feasibility / temperature sense of "collapse" (a
soft indicator hardening to a 0/1 step), distinct from the **founding bias collapse** -- the
multi-bias ``delta -> 0`` limit of an ``OMBU`` to the **closed-form** derivative
``sigma^(K-1)`` (see ``docs/theory.md``).

The relaxation layers need a ``torch`` / ``jax`` backend (import :mod:`omnibias.logic.torch`
or :mod:`omnibias.logic.jax`); the count enclosure and the oracle are pure Python.

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

from omnibias.core.proof import (
    Conjecture,
    ProofMachine,
    Verdict,
    check_certificate,
    verify_certificate_digest,
)
from omnibias.discrete import (
    AnnealSchedule,
    DiscreteSolution,
    GapCertificate,
    brute_force_min,
    certify_gap,
    decode,
)
from omnibias.discrete.maxsat import Clause, MaxSATProblem, WeightedCNF, max_sat
from omnibias.logic.model_count import (
    CountCertificate,
    CountResult,
    ModelCountProblem,
    XORClause,
    count,
    count_enclosure,
    count_models_exact,
    count_prover,
    exact_model_count,
    model_count,
    model_count_conjecture,
    prove_model_count,
    seal_count_certificate,
    treewidth_model_count,
    xor_model_count,
)

try:
    __version__ = _pkg_version("omnibias-logic")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "temperature collapse"

__all__ = [
    "AnnealSchedule",
    "Clause",
    "Conjecture",
    "CountCertificate",
    "CountResult",
    "DiscreteSolution",
    "GapCertificate",
    "MaxSATProblem",
    "ModelCountProblem",
    "ProofMachine",
    "Verdict",
    "WeightedCNF",
    "XORClause",
    "__lineage__",
    "__version__",
    "brute_force_min",
    "certify_gap",
    "check_certificate",
    "count",
    "count_enclosure",
    "count_models_exact",
    "count_prover",
    "decode",
    "exact_model_count",
    "max_sat",
    "model_count",
    "model_count_conjecture",
    "prove_model_count",
    "seal_count_certificate",
    "treewidth_model_count",
    "verify_certificate_digest",
    "xor_model_count",
]
