# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Min-square-cover as geometric optimisation: a two-register omnibias example.

Cover every 1-pixel of a binary image with the fewest fixed-size axis-aligned squares -- a
hard discrete set-cover -- reformulated as a smooth energy on continuous square centers +
existence gates, using ``omnibias-shape``'s soft occupancy / soft-OR coverage fields.

Two registers run side by side:

* **continuous** -- anneal the sharpness ``beta`` while a curvature-aware optimiser
  (exact-Hessian cubic Newton / Gauss-Newton / trust-region Newton-CG, powered by the
  closed-form ``sigma^(n)`` tower) descends the coverage energy, then round to a feasible
  discrete cover;
* **certified** -- an area / LP-relaxation lower bound (``omnibias-convex``) plus exact
  feasibility and a robustness margin, giving ``ceil(lower_bound) <= optimum <= K``.

Submodules import :mod:`torch`; import them directly (e.g.
``from examples.min_square_cover.experiment import run_sweep``) so importing this package
never requires a backend.
"""

from __future__ import annotations

__all__: list[str] = []
