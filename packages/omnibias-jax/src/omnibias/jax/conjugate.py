# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX twin of the Cauchy-Hardy dictionary (theory 01-12).

``H`` is a signed permutation of coefficients: no quadrature. Line Hilbert
only; commutation needs ``alpha > 0``. Bit-identical to the torch twin in
float64.
"""

from __future__ import annotations

from omnibias.core.conjugate import HardyDictionary, hardy_p_deriv_n, hardy_q_deriv_n

import jax.numpy as jnp
from jax import Array


def hardy_atoms(y: Array, dictionary: HardyDictionary) -> Array:
    """Evaluate every atom at ``y``. Result shape ``(*y.shape, n_atoms)``."""
    y_list = [float(v) for v in y.reshape(-1).tolist()]
    n_atoms = len(dictionary.atoms)
    flat: list[list[float]] = []
    for yi in y_list:
        row: list[float] = []
        for atom in dictionary.atoms:
            if atom.parity == "even":
                row.append(hardy_p_deriv_n(yi, atom.scale, atom.exponent, atom.order))
            else:
                row.append(hardy_q_deriv_n(yi, atom.scale, atom.exponent, atom.order))
        flat.append(row)
    out = jnp.asarray(flat, dtype=y.dtype)
    return out.reshape(*y.shape, n_atoms)


def hilbert_coeffs(coeffs: Array, dictionary: HardyDictionary) -> Array:
    """Apply the signed permutation along the last axis. No quadrature."""
    perm = dictionary.hilbert_permutation()
    n = int(coeffs.shape[-1])
    if n != len(perm):
        raise ValueError("coeffs last axis must match the dictionary")
    pieces = []
    zeros = jnp.zeros(coeffs.shape[:-1], dtype=coeffs.dtype)
    for j in range(n):
        acc = zeros
        for i, (target, sign) in enumerate(perm):
            if target == j:
                acc = acc + jnp.asarray(sign, dtype=coeffs.dtype) * coeffs[..., i]
        pieces.append(acc)
    return jnp.stack(pieces, axis=-1)


def evaluate_expansion(coeffs: Array, y: Array, dictionary: HardyDictionary) -> Array:
    """``sum_k c_k atom_k(y)``."""
    atoms = hardy_atoms(y, dictionary)
    return (atoms * coeffs).sum(axis=-1)


__all__ = ["evaluate_expansion", "hardy_atoms", "hilbert_coeffs"]
