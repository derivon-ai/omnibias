# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""JAX twin of the time-scale delta derivative of an activation.

Bit-identical mirror of :mod:`omnibias.timescale.torch`, ``jit`` / ``vmap`` / ``grad``-safe
(``name`` and the ``TimeScale`` are static, only ``z`` is traced). ``delta_derivative`` is
the closed-form ``sigma'(z)`` on ``R``, the forward difference on ``hZ``, and the Jackson
quotient on the quantum scale; ``delta_derivative_limit`` is the closed-form ``sigma'(z)`` --
the exact ``mu -> 0`` founding limit.
"""

from __future__ import annotations

from jax import Array
from omnibias.jax.activations import get_activation
from omnibias.timescale._core.timescale import TimeScale


def delta_derivative(name: str, z: Array, ts: TimeScale) -> Array:
    r"""Batched delta derivative of activation ``name`` on the time scale ``ts``."""
    spec = get_activation(name)
    if ts.kind == "reals":
        return delta_derivative_limit(name, z)
    if ts.kind == "h_integers":
        h = ts.h
        return (spec.forward(z + h) - spec.forward(z)) / h
    if ts.kind == "quantum":
        q = ts.q
        return (spec.forward(q * z) - spec.forward(z)) / ((q - 1.0) * z)
    raise ValueError(f"tensor delta_derivative supports reals/h_integers/quantum, not {ts.kind!r}")


def delta_derivative_limit(name: str, z: Array) -> Array:
    r"""Closed-form ``sigma'(z)`` via the backend fastpath -- the exact ``mu -> 0`` limit."""
    spec = get_activation(name)
    if spec.fastpath is None:
        raise NotImplementedError(f"activation {name!r} has no closed-form derivative kernel")
    out: Array = spec.fastpath(z, 1)
    return out


def delta_derivative_residual(name: str, z: Array, ts: TimeScale) -> Array:
    r"""``delta_derivative - delta_derivative_limit``; ``-> 0`` as ``mu -> 0``."""
    return delta_derivative(name, z, ts) - delta_derivative_limit(name, z)


__all__ = [
    "delta_derivative",
    "delta_derivative_limit",
    "delta_derivative_residual",
]
