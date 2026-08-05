# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias-binary: closed-form quantization gradients.

Hard forward quantizers (binary / ternary / k-bit uniform) paired with
backward passes that use the exact derivative of a smooth ``tanh(beta * z)``
surrogate via the Riccati identity ``tanh'(z) = 1 - tanh(z)^2``. Coefficients
come from :func:`omnibias.core.polynomials.tanh_polynomial_coeffs`.

Backend ops live under ``omnibias.binary.torch`` and ``omnibias.binary.jax``.

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

from omnibias.binary.schedule import BetaAnnealScheduler

try:
    __version__ = _pkg_version("omnibias-binary")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "both"

__all__ = [
    "BetaAnnealScheduler",
    "__lineage__",
    "__version__",
]
