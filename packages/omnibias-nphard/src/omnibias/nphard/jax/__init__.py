# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""JAX differentiable relaxation + decision-focused layers for omnibias-nphard.

Bit-identical (float64) twin of :mod:`omnibias.nphard.torch`. :func:`relax` is a thin
wrapper over :func:`omnibias.qubo.jax.qubo_relaxation` on ``problem.to_qubo()``;
:func:`qap_decision_cost` builds the QAP QUBO from a *predicted* flow array and returns
the true QAP objective of the relaxed decision, so a flow model trains *through* the
solver.
"""

from __future__ import annotations

from omnibias.nphard.jax.decision_focused import qap_decision_cost
from omnibias.nphard.jax.relaxation import relax

__all__ = ["qap_decision_cost", "relax"]
