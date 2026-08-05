# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable NP-hard relaxation layer (torch): a thin wrapper over the QUBO relaxation.

Bit-identical twin of :mod:`omnibias.nphard.jax.relaxation` (float64). :func:`relax`
reduces an NP-hard family problem to its QUBO (``problem.to_qubo()``) and hands it to
:func:`omnibias.qubo.torch.qubo_relaxation`, which descends a soft assignment
``x = sigmoid(beta theta) in (0, 1)^n`` along a geometric ``beta`` homotopy, *unrolled*
for backprop; as ``beta`` grows the soft assignment collapses onto a binary vertex. The
returned heatmap is decoded with :func:`omnibias.nphard.decode` and the gap certified
with :func:`omnibias.nphard.certify_gap`; to train a model that *predicts* the family
weights, use :func:`omnibias.nphard.torch.qap_decision_cost` (backprop through the solver).

Terminology: the ``beta -> inf`` hardening of ``sigmoid`` here is the feasibility /
temperature sense of "collapse" (a soft indicator becoming a 0/1 step), distinct from the
**founding bias collapse** (the multi-bias ``delta -> 0`` limit to the closed-form
derivative ``sigma^(K-1)``; see :mod:`omnibias.torch.unit` and ``docs/theory.md``).
"""

from __future__ import annotations

from typing import Any

from omnibias.qubo.problem import AnnealSchedule
from omnibias.qubo.torch import qubo_relaxation
from torch import Tensor


def relax(problem: Any, schedule: AnnealSchedule | None = None) -> Tensor:
    r"""Differentiable annealed relaxation of an NP-hard family problem -> soft ``x in (0, 1)^n``.

    ``problem`` is a :class:`~omnibias.nphard.QAPProblem` /
    :class:`~omnibias.nphard.GAPProblem` / :class:`~omnibias.nphard.SchedulingProblem` (or
    any object exposing ``to_qubo()``). The soft assignment is over the QUBO variable
    space (for QAP a flattened ``dim x dim`` matrix; reshape before decoding).
    """
    result: Tensor = qubo_relaxation(problem.to_qubo(), schedule=schedule)
    return result


__all__ = ["relax"]
