# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Lie-algebra tensor ops (jax): structure constants, bracket, trace, matrices."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from omnibias.geometry.gauge._core import kernels
from omnibias.geometry.gauge._core.lie_algebra import LieAlgebra

Array = Any


def _complex_dtype(dtype: Any) -> Any:
    return jnp.complex128 if np.dtype(dtype) == np.float64 else jnp.complex64


def structure_constants(algebra: LieAlgebra, *, dtype: Any = None) -> Array:
    """Totally antisymmetric ``f^{abc}`` as a ``(dim, dim, dim)`` real array."""
    return jnp.asarray(algebra.structure_constants(), dtype=dtype)


def symmetric_constants(algebra: LieAlgebra, *, dtype: Any = None) -> Array:
    """Totally symmetric ``d^{abc}`` as a ``(dim, dim, dim)`` real array."""
    return jnp.asarray(algebra.symmetric_constants(), dtype=dtype)


def generators(algebra: LieAlgebra, *, dtype: Any = None) -> Array:
    """The Hermitian generators ``T^a`` as a complex ``(dim, N, N)`` array."""
    real_dt = jnp.asarray(0.0).dtype if dtype is None else dtype
    return jnp.asarray(algebra.generators(), dtype=_complex_dtype(real_dt))


def bracket(x: Array, y: Array, algebra: LieAlgebra) -> Array:
    r"""Component Lie bracket ``[x, y]^c = f^{abc} x^a y^b`` for ``(B, n)`` inputs."""
    f = structure_constants(algebra, dtype=x.dtype)
    return kernels.lie_bracket(jnp, x, y, f)


def trace_product(x: Array, y: Array) -> Array:
    r"""``tr(X Y) = (1/2) x^a y^a`` in the fundamental normalization -> ``(B,)``."""
    return 0.5 * jnp.einsum("Ba,Ba->B", x, y)


def to_matrix(x: Array, algebra: LieAlgebra) -> Array:
    r"""Map adjoint components ``x^a`` to fundamental matrices ``x^a T^a``."""
    gen = generators(algebra, dtype=x.dtype)
    return kernels.to_matrix(jnp, x.astype(gen.dtype), gen)


def from_matrix(mat: Array, algebra: LieAlgebra) -> Array:
    r"""Project fundamental matrices back to real adjoint components ``2 Re tr(T^a X)``."""
    gen = generators(algebra)
    return kernels.from_matrix(jnp, mat, gen)


__all__ = [
    "bracket",
    "from_matrix",
    "generators",
    "structure_constants",
    "symmetric_constants",
    "to_matrix",
    "trace_product",
]
