# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Backend-neutral finite-difference stencils (the founding bias-collapse signs).

These are the signs of the **founding bias collapse**: ``K = order + 1`` biases on
a difference stencil (spread ``delta``) with

.. math::

    s_j = (-1)^{m-j}\,\binom{m}{j}\,/\,\delta^{m}, \qquad m = \text{order},

so that ``sum_j s_j sigma(z + b_j) -> sigma^(m)(z + b_mean)`` as ``delta -> 0``:
the biases coalesce and the finite difference *becomes* the ``m``-th derivative.
This ``delta -> 0`` limit yields a smooth ``sigma^(K-1)`` derivative -- it is
**not** the ``beta -> inf`` feasibility penalty of ``omnibias-convex`` /
``-control`` / ``-routing`` (do not conflate the two senses).

The signs and offsets are produced here as plain Python ``float`` / ``Fraction``
tuples so the :mod:`omnibias.difference.torch` and :mod:`omnibias.difference.jax`
twins materialise **the same** numbers into tensors -- bit-identical stencils by
construction, the same discipline as the shared derivative-tower coefficients.

Two stencils, matching :mod:`omnibias.torch.stencil`:

* **forward** -- offsets ``(0, delta, ..., m*delta)``; accuracy ``O(delta)``.
* **central** -- symmetric offsets ``((j - m/2)*delta)_{j=0..m}`` (mean 0);
  accuracy ``O(delta^2)`` because the odd error moments cancel.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

Stencil = str  # "forward" | "central"

_STENCILS = ("forward", "central")


def accuracy_order(stencil: Stencil) -> int:
    """The truncation-error order ``p``: ``1`` for forward, ``2`` for central."""
    if stencil == "forward":
        return 1
    if stencil == "central":
        return 2
    raise ValueError(f"unknown stencil {stencil!r}; expected one of {_STENCILS}")


def _validate(order: int, delta: float) -> None:
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if order >= 1 and delta <= 0.0:
        raise ValueError(f"delta must be > 0 for order >= 1 (got delta={delta}, order={order})")


def signs_exact(order: int, delta: Fraction) -> tuple[Fraction, ...]:
    r"""Exact ``Fraction`` finite-difference signs ``s_j = (-1)^{m-j} C(m,j)/delta^m``.

    ``order = 0`` returns ``(1,)`` (the identity single-bias unit); ``delta`` is
    then ignored.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if order == 0:
        return (Fraction(1),)
    if delta <= 0:
        raise ValueError(f"delta must be > 0 for order >= 1 (got delta={delta})")
    inv = Fraction(1) / (Fraction(delta) ** order)
    return tuple(Fraction((-1) ** (order - j)) * comb(order, j) * inv for j in range(order + 1))


def offsets_exact(order: int, delta: Fraction, stencil: Stencil = "central") -> tuple[Fraction, ...]:
    """Exact ``Fraction`` bias offsets ``b_j`` for the chosen stencil."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    d = Fraction(delta)
    if stencil == "forward":
        return tuple(Fraction(j) * d for j in range(order + 1))
    if stencil == "central":
        half = Fraction(order, 2)
        return tuple((Fraction(j) - half) * d for j in range(order + 1))
    raise ValueError(f"unknown stencil {stencil!r}; expected one of {_STENCILS}")


def stencil_signs(order: int, delta: float, stencil: Stencil = "central") -> tuple[float, ...]:
    """Finite-difference signs as ``float`` (magnitudes are stencil-independent)."""
    _validate(order, delta)
    if stencil not in _STENCILS:
        raise ValueError(f"unknown stencil {stencil!r}; expected one of {_STENCILS}")
    return tuple(float(s) for s in signs_exact(order, Fraction(delta)))


def stencil_offsets(order: int, delta: float, stencil: Stencil = "central") -> tuple[float, ...]:
    """Bias offsets ``b_j`` as ``float`` for the chosen stencil."""
    _validate(order, delta)
    return tuple(float(b) for b in offsets_exact(order, Fraction(delta), stencil))


__all__ = [
    "Stencil",
    "accuracy_order",
    "offsets_exact",
    "signs_exact",
    "stencil_offsets",
    "stencil_signs",
]
