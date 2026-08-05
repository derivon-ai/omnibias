# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias-score: score-based / SDE operators.

The closed-form score ``grad log p``, the Ito infinitesimal generator
``L f = b . grad f + 1/2 tr(a hess f)``, and the Fokker-Planck adjoint
``L* p``, composed from the ``omnibias-fields`` closed-form gradient / Hessian
primitives (no new low-level kernels).

Backend ops live under ``omnibias.score.torch`` and ``omnibias.score.jax``.

The :mod:`omnibias.score.flow` submodule (**alpha**, folded in from the former
standalone ``omnibias-flow`` package) adds continuous normalizing flows with an
exact ``omnibias-fields`` trace-of-Jacobian -- the CNF companion to these
score / SDE operators in one probability-flow package.

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

try:
    __version__ = _pkg_version("omnibias-score")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = ["__lineage__", "__version__"]
