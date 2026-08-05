# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable annealed #SAT model-finder relaxation (JAX).

Bit-identical twin of :mod:`omnibias.logic.torch.relaxation` (float64). For #SAT every clause
is **hard**, so *finding a satisfying assignment* is exactly *driving the weighted-violation
energy to zero*. :func:`sat_relaxation` therefore reuses the MaxSAT annealed relaxation on the
instance's internal hard-clause ``MaxSATProblem`` -- a thin front-end over the shared
:func:`omnibias.discrete.jax.anneal_descent`, which anneals ``x = sigmoid(beta theta)`` onto a
binary vertex as ``beta -> inf`` (unrolled so a model predicting the formula can train
*through* it). Decoding the soft output yields a model iff its energy is ``0``; distinct
decoded models are sound witness lower bounds for :func:`omnibias.logic.count_enclosure`.

Terminology: the ``beta -> inf`` hardening of ``sigmoid`` here is the feasibility /
temperature sense of "collapse" (a soft indicator becoming a 0/1 step), distinct from the
**founding bias collapse** (the multi-bias ``delta -> 0`` limit to ``sigma^(K-1)``, a
derivative; see ``docs/theory.md`` and :mod:`omnibias.torch.unit`).
"""

from __future__ import annotations

from jax import Array
from omnibias.discrete._core.schedule import AnnealSchedule
from omnibias.discrete.maxsat.jax import maxsat_relaxation
from omnibias.logic.model_count.problem import ModelCountProblem


def sat_relaxation(problem: ModelCountProblem, schedule: AnnealSchedule | None = None) -> Array:
    r"""Differentiable annealed model-finder relaxation of a #SAT instance -> ``x in (0, 1)^n``.

    Descends the closed-form hard-clause violation-energy gradient while ``beta -> inf``
    anneals the soft assignment onto a binary vertex; the result decodes to a model when the
    instance is satisfiable.
    """
    result: Array = maxsat_relaxation(problem.as_maxsat, schedule)
    return result


__all__ = ["sat_relaxation"]
