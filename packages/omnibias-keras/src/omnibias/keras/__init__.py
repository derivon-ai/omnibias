# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias.keras: Keras 3 unified backend for omnibias.

The same code runs on TensorFlow, JAX, or PyTorch because every kernel is
written against :mod:`keras.ops`. Closed-form n-th derivative towers
(``sigma^(n)(z)``) share their polynomial coefficients with the torch and
JAX backends via :mod:`omnibias.core.polynomials`, so the closed-form
activation / derivative math is bit-identical across backends by construction.
End-to-end layer numerics otherwise follow the selected Keras backend
(TensorFlow / JAX / PyTorch).

Select the Keras backend with the ``KERAS_BACKEND`` environment variable
(``tensorflow`` | ``jax`` | ``torch``) *before* importing ``keras``.

Public API:

- :class:`OperatorMultiBiasUnit` (alias :class:`OMBU`): trainable
  scalar-operator primitive (a ``keras.layers.Layer``).
- :class:`GrowableOperatorMultiBiasUnit` (alias :class:`GrowableOMBU`):
  the OMBU with a growable bias arity ``K``.
- :class:`OperatorBlock`: typed wrapper selecting K and forward behaviour
  from ``op="grad"|"laplacian"|"derivative"|"band"|"integral"|"identity"``.
- :class:`cmbDense`, :class:`cmbConv1D`, :class:`cmbConv2D`: drop-in
  ``Dense`` / ``Conv1D`` / ``Conv2D`` with an inline :class:`OperatorBlock`.
- :class:`KGrowthScheduler`: plateau-triggered K-growth controller.
- :func:`get_activation`, :func:`list_activations`,
  :func:`register_activation`, :func:`is_registered`: registry accessors.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("omnibias-keras")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

from omnibias.keras.activations import (
    get_activation,
    is_registered,
    list_activations,
    register_activation,
)
from omnibias.keras.activations.registry import ActivationSpec
from omnibias.keras.blocks import (
    AnalyticGaussianConv1D,
    AnalyticGaussianConv2D,
    OperatorBlock,
    OpName,
    analytic_gaussian_taps,
    cmbConv1D,
    cmbConv2D,
    cmbDense,
)
from omnibias.keras.growable import GrowableOperatorMultiBiasUnit, GrowStrategy
from omnibias.keras.tempered_blocks import LearnablePReLU, TemperedActivation
from omnibias.keras.training import KGrowthScheduler
from omnibias.keras.unit import OperatorMultiBiasUnit

OMBU = OperatorMultiBiasUnit
GrowableOMBU = GrowableOperatorMultiBiasUnit

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = [
    "ActivationSpec",
    "AnalyticGaussianConv1D",
    "AnalyticGaussianConv2D",
    "GrowStrategy",
    "GrowableOMBU",
    "GrowableOperatorMultiBiasUnit",
    "KGrowthScheduler",
    "LearnablePReLU",
    "OMBU",
    "OpName",
    "OperatorBlock",
    "OperatorMultiBiasUnit",
    "TemperedActivation",
    "__lineage__",
    "__version__",
    "analytic_gaussian_taps",
    "cmbConv1D",
    "cmbConv2D",
    "cmbDense",
    "get_activation",
    "is_registered",
    "list_activations",
    "register_activation",
]
