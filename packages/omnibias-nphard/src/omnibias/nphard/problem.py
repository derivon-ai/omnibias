# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-agnostic containers for omnibias-nphard.

The three NP-hard families (:class:`~omnibias.nphard.QAPProblem`,
:class:`~omnibias.nphard.GAPProblem`, :class:`~omnibias.nphard.SchedulingProblem`) live
in :mod:`omnibias.nphard._core`; each implements the ``omnibias-discrete``
``DiscreteProblem`` seam and exposes ``to_qubo()`` so the shared ``omnibias-qubo``
relaxation / decoder / certificate apply. This module only re-exports the shared
schedule / certificate containers so ``omnibias.nphard`` presents one coherent surface.

* :data:`AnnealSchedule` -- the ``beta -> inf`` homotopy that drives the annealed
  relaxation (owned by the discrete substrate, re-exported here).
* :data:`NPHardCertificate` -- an alias of the discrete substrate's ``GapCertificate``:
  a **gap-shaped** container ``lower_bound <= optimum <= energy``. The gap can be
  non-tight; the container has no ``is_optimal`` / ``is_exact`` field. A rigorously
  closed gap would prove this instance's optimality, without implying a polynomial
  worst-case algorithm or P = NP.

Terminology: the relaxation that consumes these containers hardens
``sigmoid(beta z)`` as ``beta -> inf`` -- the feasibility / temperature sense of
"collapse", distinct from the **founding bias collapse** (the multi-bias
``delta -> 0`` limit to the closed-form derivative ``sigma^(K-1)``; see
``docs/theory.md``).
"""

from __future__ import annotations

from omnibias.discrete import AnnealSchedule, GapCertificate

# The rigorous optimality-gap certificate is the discrete substrate's gap-shaped
# container. Its gap is generally non-tight; instance optimality and worst-case
# algorithmic complexity are separate claims.
NPHardCertificate = GapCertificate

__all__ = [
    "AnnealSchedule",
    "NPHardCertificate",
]
