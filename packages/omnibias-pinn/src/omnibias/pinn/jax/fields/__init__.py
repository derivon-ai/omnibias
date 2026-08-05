# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Typed PINN fields for the JAX backend.

Each field is a :class:`equinox.Module`-style frozen dataclass
(implemented here as a plain ``dataclass(frozen=True)`` carrying
:class:`jax.Array` parameters) whose ``__call__(coords)`` returns a
:class:`omnibias.pinn.FieldState`. The state primes the lazy
:class:`SigmaCache`, exposes the :class:`ComponentSpec` /
:class:`CoordinateSpec` metadata, and routes attribute access into the
JAX ops dispatch module.

JAX-specific note: we do *not* require equinox. The fields are vanilla
frozen dataclasses with ``jax.Array`` parameters. Users that want
trainable fields can use :func:`jax.tree_util.register_pytree_node`
themselves (the pytree registration is provided by the field class
constructor automatically).
"""

from __future__ import annotations

from omnibias.pinn.jax.fields.base import FieldBase
from omnibias.pinn.jax.fields.chebyshev import (
    ChebyshevVectorField,
    make_chebyshev_vector_field,
)
from omnibias.pinn.jax.fields.one_layer import (
    OneLayerVectorField,
    make_one_layer_vector_field,
)
from omnibias.pinn.jax.fields.spectral import (
    SpectralVectorField,
    make_spectral_vector_field,
)

__all__ = [
    "ChebyshevVectorField",
    "FieldBase",
    "OneLayerVectorField",
    "SpectralVectorField",
    "make_chebyshev_vector_field",
    "make_one_layer_vector_field",
    "make_spectral_vector_field",
]
