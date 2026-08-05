# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""omnibias-convex: differentiable + certified convex optimization (LP/QP).

A closed-form-Hessian log-barrier interior-point solver, an ``argmin`` that is
differentiable through the KKT system (OptNet / cvxpylayers style), and an
optional rigorous optimality certificate from :mod:`omnibias.core.verified`.

Backend solvers live under ``omnibias.convex.jax`` and ``omnibias.convex.torch``;
the pure containers (:class:`ConvexSolution`, :class:`BarrierOptions`) are shared.

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

from omnibias.convex.certify import (
    Certificate,
    CertificationError,
    certify_lp_optimum,
    certify_qp_optimum,
    lp_dual_lower_bound,
)
from omnibias.convex.problem import BarrierOptions, ConvexSolution
from omnibias.convex.warm_start import (
    active_set_warm_start,
    geometry_warm_start,
    predicted_vertex,
)

try:
    __version__ = _pkg_version("omnibias-convex")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "temperature collapse"

__all__ = [
    "BarrierOptions",
    "Certificate",
    "CertificationError",
    "ConvexSolution",
    "__lineage__",
    "__version__",
    "active_set_warm_start",
    "certify_lp_optimum",
    "certify_qp_optimum",
    "geometry_warm_start",
    "lp_dual_lower_bound",
    "predicted_vertex",
]
