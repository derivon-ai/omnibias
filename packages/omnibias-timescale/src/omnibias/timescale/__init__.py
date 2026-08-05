# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""omnibias-timescale: time-scale (Hilger) calculus unifying continuous and discrete.

A :class:`TimeScale` (``R``, ``hZ``, the quantum ``q^Z``, or a finite set) carries the jump
operators ``sigma`` / ``rho`` and the graininess ``mu``. The delta derivative
:func:`delta_derivative` / :func:`delta_derivative_tower` is one operator over three
registers: the **closed-form omnibias tower** on ``R``, the :mod:`omnibias.difference`
forward difference on ``hZ``, and the :mod:`omnibias.qcalculus` Jackson q-derivative on the
quantum scale. On top sit the delta integral, the Hilger exponential with its
``circle-plus`` regressive group, and linear dynamic equations.

This is the **founding bias collapse** generalized to a variable mesh: omnibias's founding
move is the ``delta -> 0`` limit collapsing a many-bias finite difference into the smooth
derivative ``sigma^(K-1)``; here the graininess ``mu`` *is* that ``delta``, and as
``mu -> 0`` the delta derivative collapses to the ordinary derivative. It is the derivative
sense of "collapse" -- **not** the ``beta -> inf`` feasibility penalty -- and is distinct
from the ``q -> 1`` limit of :mod:`omnibias.qcalculus` (fixed mesh, varying deformation).

Backend twins (:mod:`omnibias.timescale.torch`, :mod:`omnibias.timescale.jax`) provide the
bit-identical delta derivative of the activation dictionary; import them explicitly.

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

from omnibias.timescale._core import (
    TimeScale,
    circle_minus,
    circle_plus,
    cylinder,
    delta_derivative,
    delta_derivative_tower,
    delta_integral,
    finite,
    h_integers,
    hilger_exponential,
    is_regressive,
    nabla_derivative,
    quantum,
    reals,
    sigma_value,
    solve_linear_dynamic,
    variation_of_constants,
)

try:
    __version__ = _pkg_version("omnibias-timescale")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = [
    "TimeScale",
    "__lineage__",
    "__version__",
    "circle_minus",
    "circle_plus",
    "cylinder",
    "delta_derivative",
    "delta_derivative_tower",
    "delta_integral",
    "finite",
    "h_integers",
    "hilger_exponential",
    "is_regressive",
    "nabla_derivative",
    "quantum",
    "reals",
    "sigma_value",
    "solve_linear_dynamic",
    "variation_of_constants",
]
