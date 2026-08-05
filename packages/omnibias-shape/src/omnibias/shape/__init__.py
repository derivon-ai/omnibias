# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""omnibias-shape: differentiable soft shape / occupancy fields + soft coverage.

A soft axis-aligned box is a product of sigmoid-pair interval indicators; a soft
*cover* is a soft-OR (or log-sum-exp) union of such shapes. Because the shapes are
built purely from ``sigmoid``, every center-derivative is a closed-form polynomial
in ``sigmoid`` via the shared Riccati tower
(:func:`omnibias.core.polynomials.sigmoid_polynomial_coeffs`), so the gradient *and*
Hessian of a coverage energy with respect to the shape centers are available in
closed form (see ``coverage_energy_grad`` / ``coverage_energy_hessian``).

This is the primitive that turns a discrete geometric-covering problem (cover every
1-pixel of a binary image with the fewest fixed-size squares) into a smooth,
second-order-optimizable energy. The operator surface lives under
``omnibias.shape.torch.ops`` and ``omnibias.shape.jax.ops`` as bit-identical twins;
the root module exports only ``__version__``.

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
    __version__ = _pkg_version("omnibias-shape")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "both"

__all__ = ["__lineage__", "__version__"]
