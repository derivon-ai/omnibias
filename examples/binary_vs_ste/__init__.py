# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Binary-neural-network training: omnibias surrogate *menu* vs STE.

A controlled, apples-to-apples benchmark. Every arm shares the *exact same* hard
``sign`` forward, architecture, initialisation, optimiser and data order; only the
**backward through the quantizer** differs -- the surrogate-gradient *kernel* and
its bandwidth ``beta``:

* ``ste`` -- the straight-through estimator (compact box kernel, ``beta=1``), the
  classic BNN baseline;
* ``omnibias_b10`` -- the shipped ``binarize`` (exact ``beta * tanh'(beta z)``) at the
  library default ``beta=10`` -- the *mis-scaled* control whose narrow gradient
  window starves units;
* ``omnibias_b1`` -- the same shipped ``binarize`` at the correctly scaled ``beta=1``;
* ``tanh`` / ``logistic`` / ``gaussian`` / ``cauchy`` -- the single-hyperplane kernel
  menu, peak-normalised so ``beta`` sets only the window *width*. ``cauchy``'s heavy
  tails keep far-from-boundary units alive (no dead units);
* ``anneal`` -- the ``tanh`` kernel with a soft-to-hard ``beta`` curriculum (0.5 -> 3);
* ``learnable_beta`` -- the ``tanh`` kernel with a trained bandwidth ``beta``.

The forward is always the exact hard quantizer, so this measures *which surrogate
gradient trains better*; it is not a claim that the hard step is differentiable.
The key lesson: ``beta`` must match the (post-BatchNorm) activation scale -- a too
sharp surrogate (large ``beta``, or the STE box) drops the gradient window below the
data and starves most units.

Submodules import :mod:`torch`; import them directly (e.g.
``from examples.binary_vs_ste.experiment import run_sweep``) so that importing
this package never requires a deep-learning backend.
"""

from __future__ import annotations

__all__: list[str] = []
