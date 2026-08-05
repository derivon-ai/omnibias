# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Complex-valued activations for symmetry-projected / k-point NQS.

Real-valued wavefunctions can only represent ground states at the
high-symmetry k-points ``Γ = (0,0)`` and ``M = (π,π)`` (where the
representation is real). For all other k-points — including the
``X = (π,0)`` and ``Y = (0,π)`` momenta relevant to certain stripe
ordering directions — the wavefunction must be complex.

This module provides the canonical complex-valued NQS activations:

* ``modrelu(z, b)`` — ``(|z| + b) · z / |z|`` for ``|z| + b ≥ 0``,
  else 0. The Arjovsky et al. 2016 unitary-RNN activation. Smooth
  variant uses :func:`softabs` instead of ``|z|`` to maintain
  differentiability through ``z = 0``.

* ``cardioid(z)`` — ``0.5 · (1 + cos(arg(z))) · z``. Real-valued at
  real ``z``, smooth everywhere except possibly ``z = 0``.

Both activations are written as functions of ``z ∈ C``; the JAX
autodiff machinery handles real and imaginary parts as a vector
``(Re z, Im z) ∈ R²``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

__all__ = [
    "cardioid",
    "complex_abs2",
    "complex_phase",
    "modrelu",
    "softmodrelu",
]


def complex_abs2(z: Array) -> Array:
    """Squared modulus ``|z|²`` for complex ``z``."""
    return (z.real * z.real + z.imag * z.imag) if jnp.iscomplexobj(z) else z * z


def complex_phase(z: Array, eps: float = 1e-12) -> Array:
    """Argument ``arg(z)`` in ``[-π, π]``. Robust to ``z = 0`` via ``eps``."""
    if jnp.iscomplexobj(z):
        return jnp.arctan2(z.imag, z.real + eps)
    return jnp.zeros_like(z)


def modrelu(z: Array, b: float = 0.0) -> Array:
    r"""Modular ReLU (Arjovsky et al., 2016).

    .. math::

        \text{modrelu}(z, b) = \begin{cases}
            (|z| + b) \cdot z / |z|  & \text{if } |z| + b \ge 0 \\
            0                         & \text{otherwise}
        \end{cases}

    The bias ``b`` controls a soft magnitude threshold: outputs are
    zero when the magnitude drops below ``-b``. With ``b = 0`` this
    reduces to the identity, which is rarely useful — typical ``b``
    values are negative (e.g., ``-1``).

    Non-differentiable at ``z = 0``; for a smooth variant use
    :func:`softmodrelu`.
    """
    mag = jnp.abs(z)
    out = jnp.where(mag + b >= 0.0, (mag + b) * z / (mag + 1e-30), jnp.zeros_like(z))
    return out


def softmodrelu(z: Array, b: float = 0.0, eps: float = 1e-3) -> Array:
    r"""Smooth modular ReLU using :func:`softabs` for the magnitude.

    Replaces ``|z|`` with ``softabs(|z|) = sqrt(|z|² + eps²) - eps``
    so that the activation is differentiable through ``z = 0``. The
    output approaches :func:`modrelu` as ``eps → 0``.
    """
    abs_z = jnp.sqrt(complex_abs2(z) + eps * eps) - eps
    return jnp.where(
        abs_z + b >= 0.0,
        (abs_z + b) * z / (abs_z + eps),
        jnp.zeros_like(z),
    )


def cardioid(z: Array) -> Array:
    r"""Cardioid activation: ``0.5 · (1 + cos(arg(z))) · z``.

    Named for the cardioid-shaped magnitude profile in the complex plane
    (Trabelsi et al. 2018). It reduces to ``z`` for real positive ``z``,
    ``0`` for real negative ``z``, and varies smoothly elsewhere with
    a clean gradient through ``z = 0``.

    Useful as a complex analogue of ReLU in self-attention residual
    paths for complex-valued NQS.
    """
    phi = complex_phase(z)
    return 0.5 * (1.0 + jnp.cos(phi)) * z
