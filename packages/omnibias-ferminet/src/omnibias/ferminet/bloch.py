# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Bloch / twist mixed partials on a periodic cell via multivariate jets.

A real observable on a 1-D cell with a twist angle ``theta`` is the
activation of an affine form ``w_x x + w_theta theta``.  The mixed partial
``d^2 f / dx d theta`` is read off a single :func:`omnibias.jax.jet_mv.layer_jet_mv`
pass -- the primitive contract already permits mixed partials, so this module
does not invent a new jet kernel.

Complex Bloch phases ``e^{i theta x}`` are **not** in the real activation
tower; this is a real twist observable, not a holomorphic band structure.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.jax.jet_mv import identity_jet, jet_partials, layer_jet_mv


def bloch_twist_mixed_partials(
    x: Array,
    twist: Array,
    *,
    weight_x: float = 1.0,
    weight_twist: float = 1.0,
    bias: float = 0.0,
    spec: str = "sigmoid",
    order: int = 2,
) -> dict[tuple[int, ...], Array]:
    """Mixed partials of ``sigma(w_x x + w_theta twist + b)`` up to ``order``.

    Coordinates are ``(x, twist)``.  The Hessian entry ``(1, 1)`` in
    multi-index language is ``d^2 / dx d twist``.
    """
    point = jnp.stack([jnp.asarray(x), jnp.asarray(twist)])
    jet = identity_jet(point, order)
    weights = jnp.asarray([[float(weight_x), float(weight_twist)]], dtype=point.dtype)
    b = jnp.asarray([float(bias)], dtype=point.dtype)
    out = layer_jet_mv(jet, weights, b, spec, dim=2, order=order)
    # layer output is shape (M, 1); squeeze the unit axis for scalar partials
    squeezed = out[:, 0]
    return jet_partials(squeezed, dim=2, order=order)


__all__ = ["bloch_twist_mixed_partials"]
