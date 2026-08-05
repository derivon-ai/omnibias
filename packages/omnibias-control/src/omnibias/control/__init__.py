# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""omnibias-control: differentiable control with a model-relative safety certificate.

A batched, per-sample control-barrier-function (CBF-QP) **safety filter** -- the
temperature-collapse projection layer specialised to state-dependent constraints -- with:

* autodiff Lie-derivative **CBF-row builders** for any control-affine system
  ``x_dot = f(x) + g(x) a`` (:func:`~omnibias.control.jax.builders.control_affine_cbf_rows`)
  and for dynamics produced by a (possibly learned) **Lagrangian**
  (:func:`~omnibias.control.jax.builders.lagrangian_cbf_rows`, reusing
  :mod:`omnibias.variational`);
* a differentiable **safe rollout** so a policy can be trained *through* the filter;
* a rigorous **model-relative recoverable-set certificate**
  (:func:`~omnibias.control.certify.certify_recoverable`, via :mod:`omnibias.verify`).

Backend solvers live under ``omnibias.control.jax`` and ``omnibias.control.torch``
(bit-identical twins); the pure containers and the certificate are shared.

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

from omnibias.control.certify import (
    certify_disc_recoverable,
    certify_recoverable,
    disc_obstacle_margin,
)
from omnibias.control.problem import (
    CBFSpec,
    FilterSchedule,
    RecoverableCertificate,
    SafeAction,
)

try:
    __version__ = _pkg_version("omnibias-control")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "temperature collapse"

__all__ = [
    "CBFSpec",
    "FilterSchedule",
    "RecoverableCertificate",
    "SafeAction",
    "__lineage__",
    "__version__",
    "certify_disc_recoverable",
    "certify_recoverable",
    "disc_obstacle_margin",
]
