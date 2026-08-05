# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable Boolean spectrum engine (jax) via multivariate jets.

Bit-identical twin of :mod:`omnibias.boolean.torch.ops.spectrum` (double precision
required for exact cross-backend parity -- enable ``jax_enable_x64``). The
multilinear extension is built as a multivariate jet with
:func:`omnibias.jax.jet_mv.identity_jet` and the truncated Cauchy product, and the
Mobius / Walsh spectra are read off with
:func:`omnibias.jax.jet_mv.jet_partials`.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.core.multi_index import multiply_table
from omnibias.jax.jet_mv import identity_jet, jet_partials


def _num_vars(size: int) -> int:
    if size < 1 or (size & (size - 1)) != 0:
        raise ValueError(f"values length must be a power of two, got {size}")
    return size.bit_length() - 1


def _as_array(values: Sequence[float] | Array) -> Array:
    return jnp.asarray(values)


def _mul_jet(
    p: Array, q: Array, g_idx: Array, a_idx: Array, b_idx: Array, m: int
) -> Array:
    """Truncated multivariate polynomial product via gather + segment scatter-add."""
    contrib = p[a_idx] * q[b_idx]
    return jnp.zeros(m, dtype=p.dtype).at[g_idx].add(contrib)


def _extension_jet(values: Array, mode: str, order: int) -> tuple[Array, int]:
    n = _num_vars(values.shape[0])
    x0 = jnp.zeros(n, dtype=values.dtype)
    coord = identity_jet(x0, order)
    m = coord.shape[0]
    table = multiply_table(n, order)
    g_idx = np.array([t[0] for t in table])
    a_idx = np.array([t[1] for t in table])
    b_idx = np.array([t[2] for t in table])
    one = jnp.zeros(m, dtype=values.dtype).at[0].set(1.0)
    g1: list[Array] = []
    g0: list[Array] = []
    for i in range(n):
        c = coord[:, i]
        if mode == "mobius":
            g1.append(c)
            g0.append(one - c)
        elif mode == "walsh":
            g1.append((one - c) * 0.5)
            g0.append((one + c) * 0.5)
        else:
            raise ValueError(f"unknown mode {mode!r}")
    total: Array | None = None
    for a in range(1 << n):
        prod = one
        for i in range(n):
            factor = g1[i] if (a >> i) & 1 else g0[i]
            prod = _mul_jet(prod, factor, g_idx, a_idx, b_idx, m)
        term = values[a] * prod
        total = term if total is None else total + term
    assert total is not None
    return total, n


def _coeffs_vector(values: Array, mode: str) -> Array:
    n = _num_vars(values.shape[0])
    jet, _ = _extension_jet(values, mode, order=n)
    partials = jet_partials(jet, n, n)
    out: list[Array | None] = [None] * (1 << n)
    for alpha, val in partials.items():
        if all(e <= 1 for e in alpha):
            mask = 0
            for i, e in enumerate(alpha):
                if e:
                    mask |= 1 << i
            out[mask] = val
    zero = values[0] * 0.0
    return jnp.stack([zero if x is None else x for x in out])


def mobius_coeffs(values: Sequence[float] | Array) -> Array:
    """``{0,1}`` Mobius / multilinear coefficients ``m_S`` (indexed by subset mask)."""
    return _coeffs_vector(_as_array(values), "mobius")


def walsh_coeffs(values: Sequence[float] | Array) -> Array:
    """``{+-1}`` Walsh / Fourier coefficients ``hat f(S)`` (indexed by subset mask)."""
    return _coeffs_vector(_as_array(values), "walsh")


def _to_spectrum(coeffs: Array) -> dict[frozenset[int], Array]:
    n = _num_vars(coeffs.shape[0])
    return {
        frozenset(i for i in range(n) if (mask >> i) & 1): coeffs[mask]
        for mask in range(1 << n)
    }


def mobius_spectrum(values: Sequence[float] | Array) -> dict[frozenset[int], Array]:
    """Mobius coefficients as a ``{variable-set: m_S}`` mapping."""
    return _to_spectrum(mobius_coeffs(values))


def walsh_spectrum(values: Sequence[float] | Array) -> dict[frozenset[int], Array]:
    """Walsh/Fourier coefficients as a ``{variable-set: hat f(S)}`` mapping."""
    return _to_spectrum(walsh_coeffs(values))


def influences_diff(values: Sequence[float] | Array) -> Array:
    """Per-coordinate influence ``Inf_i = sum_{S in i} hat f(S)^2`` (differentiable)."""
    w = walsh_coeffs(values)
    n = _num_vars(w.shape[0])
    out = []
    for i in range(n):
        idx = [mask for mask in range(1 << n) if (mask >> i) & 1]
        out.append((w[jnp.asarray(idx)] ** 2).sum())
    return jnp.stack(out)


def algebraic_degree_soft(values: Sequence[float] | Array) -> Array:
    """Spectral-energy-weighted mean monomial order (a smooth degree proxy)."""
    w = walsh_coeffs(values)
    n = _num_vars(w.shape[0])
    orders = jnp.asarray(
        [bin(mask).count("1") for mask in range(1 << n)], dtype=w.dtype
    )
    energy = (w**2).sum()
    return (orders * w**2).sum() / energy


__all__ = [
    "algebraic_degree_soft",
    "influences_diff",
    "mobius_coeffs",
    "mobius_spectrum",
    "walsh_coeffs",
    "walsh_spectrum",
]
