# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias-measure: autograd-native measure-theoretic integration.

This package is the third of omnibias's three distinct senses of "integral"
(see ``docs/operator-surface.md``): not the closed-form activation
antiderivative (``OperatorBlock(op="integral")``) and not the field domain
quadrature (``omnibias.fields.integrate``), but the **measure integral**
``int f dmu`` against an abstract measure.

The backend-agnostic substrate lives in :mod:`omnibias.measure._core`:

* :class:`~omnibias.measure._core.measure.Measure` -- a discrete measure
  ``(nodes, weights)`` generalizing ``omnibias.fields``'s ``QuadratureSpec``,
  with ``pushforward`` (change of variables), ``product`` (product measure),
  ``reweight`` (Radon-Nikodym / importance reweighting) and ``normalize``.
* the primitives :func:`lebesgue_integral`, :func:`importance_expectation`,
  :func:`layer_cake_integral` and :func:`simple_function_approx`.
* :mod:`~omnibias.measure._core.integraleq` -- **Fredholm and Volterra integral
  equations of the second kind** on that same quadrature. A ``Measure`` is
  already nodes and weights, which is exactly what Nystrom discretisation needs,
  so ``u = f + lam int K u dmu`` becomes the linear system
  ``(I - lam K W) u = f`` for free.

The numpy ``_core`` is the bit-identical reference; the differentiable twins
live in ``omnibias.measure.torch`` and ``omnibias.measure.jax`` (imported lazily,
so importing :mod:`omnibias.measure` pulls in neither torch nor jax) and ship
trainable ``nn.Module`` / functional layer wrappers whose measure weights and
softness ``beta`` can be learned end-to-end.

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

from omnibias.measure._core import (
    Measure,
    NeumannResult,
    SimpleFunctionApprox,
    degenerate_kernel_solve,
    fredholm_residual,
    importance_expectation,
    layer_cake_integral,
    lebesgue_integral,
    neumann_series,
    nystrom_solve,
    simple_function_approx,
    solvability_margin,
    superlevel_measure,
    volterra_solve,
)

try:
    __version__ = _pkg_version("omnibias-measure")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "temperature collapse"

__all__ = [
    "Measure",
    "NeumannResult",
    "SimpleFunctionApprox",
    "__lineage__",
    "__version__",
    "degenerate_kernel_solve",
    "fredholm_residual",
    "importance_expectation",
    "layer_cake_integral",
    "lebesgue_integral",
    "neumann_series",
    "nystrom_solve",
    "simple_function_approx",
    "solvability_margin",
    "superlevel_measure",
    "volterra_solve",
]
