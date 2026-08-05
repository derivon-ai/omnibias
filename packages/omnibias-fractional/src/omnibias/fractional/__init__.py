# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias-fractional: fractional calculus in two honest classes.

Two distinct operator classes live here:

* **Grid / spectral** (:mod:`omnibias.fractional.torch.ops.fractional` and its
  jax twin): Grunwald-Letnikov, Riemann-Liouville and Caputo derivatives on
  uniform grids, plus spectral (FFT) derivatives on periodic domains. These are
  **non-local numerical approximations** whose accuracy is set by the grid /
  spectrum -- **not** closed-form sigma-tower derivatives.
* **Analytic** (:mod:`omnibias.fractional.torch.ops.analytic` and its jax twin):
  ``fractional_derivative`` / ``mlp_fractional_derivative`` evaluate the
  closed-form gamma-ratio Taylor-jet series. This **is** closed form, but on the
  *analytic-function class* only -- an order-``N`` truncation that is exact for
  polynomials of degree ``<= N``.

See ``FRACTIONAL_DERIVATIONS.md`` for the derivations and error budget.

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
    __version__ = _pkg_version("omnibias-fractional")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = ["__lineage__", "__version__"]
