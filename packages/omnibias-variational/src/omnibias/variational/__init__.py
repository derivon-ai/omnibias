# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias-variational: the Least Action Principle for omnibias.

Hamilton's principle -- the physical trajectory makes the action
``S = integral L(q, qdot, t) dt`` stationary, ``delta S = 0``, giving the
Euler-Lagrange equations ``d/dt(dL/dqdot) - dL/dq = 0`` -- built on the
``omnibias-fields`` substrate. The trajectory / field derivatives are the
closed-form sigma-tower derivatives; the action is a quadrature of the field;
the Lagrangian's own partials are autodiff of the user callable.

The backend-agnostic schemas (:class:`Lagrangian`, :class:`LagrangianDensity`,
:class:`Constraint`) live here; the backend ops live under
``omnibias.variational.torch`` and ``omnibias.variational.jax``, and the
rigorous enclosures in ``omnibias.variational.verified``.

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

from omnibias.variational._core import (
    Constraint,
    Hamiltonian,
    Lagrangian,
    LagrangianDensity,
)

try:
    __version__ = _pkg_version("omnibias-variational")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = [
    "Constraint",
    "Hamiltonian",
    "Lagrangian",
    "LagrangianDensity",
    "__lineage__",
    "__version__",
]
