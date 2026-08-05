# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias.score.flow: continuous normalizing flows with exact trace-of-Jacobian.

For an ODE ``dx/dt = v(t, x)`` the instantaneous log-density change is
``d log p / dt = -div_x v(t, x) = -tr(partial v / partial x)``. This
package computes that divergence **exactly** via the ``omnibias-fields``
divergence operator (not a Hutchinson estimator). The trace term is exact;
the ODE time-integration itself is a standard numerical solver. The trace term is exact;
the ODE time-integration itself is a standard numerical solver.

Backend ops live under ``omnibias.score.flow.torch`` and ``omnibias.score.flow.jax``.

Maturity: this is an **alpha** submodule folded in from the former standalone
``omnibias-flow`` package; it now lives inside the ``omnibias-score``
probability-flow distribution alongside the score / SDE / Fokker-Planck ops.
"""

from __future__ import annotations

__all__: list[str] = []
