# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias.pinn.train: causal marching drivers and training diagnostics.

Closes the loop that
:class:`~omnibias.pinn._core.marching.TimeMarcher` deliberately left open:
the marcher answers *which points, which epsilon, may I advance*, and this
submodule owns the optimiser loop that consumes those answers.

Backend-free diagnostics
(:func:`~omnibias.pinn.train._core.causality.causality_index`,
:func:`~omnibias.pinn.train._core.guards.trivial_solution_guard`) live under
``_core``. Numeric drivers live under ``omnibias.pinn.train.torch`` /
``omnibias.pinn.train.jax`` and are imported explicitly so
``import omnibias.pinn.train`` never pulls in a backend.

Maturity: **alpha** submodule of the Beta ``omnibias-pinn`` distribution.

Guarantee level
---------------
Window geometry and advance gating are by construction; the causality index and
triviality guards are measurements. Training success is empirical on the named
PDE families in ``docs/benchmarks.md`` -- not an a-priori error method.
"""

from __future__ import annotations

from omnibias.pinn.train._core import (
    CausalityReport,
    SpectralBandScheduler,
    TrivialSolutionVerdict,
    causality_index,
    report_causality,
    trivial_solution_guard,
    unlocked_fraction,
)

__all__ = [
    "CausalityReport",
    "SpectralBandScheduler",
    "TrivialSolutionVerdict",
    "causality_index",
    "report_causality",
    "trivial_solution_guard",
    "unlocked_fraction",
]
