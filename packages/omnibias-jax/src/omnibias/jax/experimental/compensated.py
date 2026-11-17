# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Error-free matrix products for accurate Gram assembly.

A float64 matrix product ``A B`` with inner dimension ``P`` carries a rounding error that grows with ``P`` (up to
``P eps`` times the sum of absolute products).  The damped Gauss-Newton step multiplies the error of ``K = J J^T`` by the
condition number of ``K + d I``, so for a nearly singular ``K`` the assembly error, not the derivative program, limits
the step (``conditioning.py``).  :func:`exact_matmul` removes the assembly error: each operand is split into slices with
so few significant bits that every slice product is an integer multiple of a power of two below :math:`2^{53}`, which any
floating-point summation order and a fused multiply-add evaluate exactly; the exact partial products are then added in
order of significance.  The result is the product correct to a few units in the last place of the *result*, independent
of ``P``, at the cost of ``k (k + 1) / 2`` matrix products for ``k`` slices (10 for the default ``k = 4``).  Each slice
carries ``beta = floor((53 - log2 P) / 2)`` bits *relative to the largest entry of its row*, so an entry smaller than
that maximum by a factor ``2^w`` keeps only ``k beta - w`` bits: full float64 accuracy of every entry needs
``k beta >= 53 + w``.  Rows of parameter Jacobians span many orders of magnitude; ``k = 4`` covers ``w`` up to about 30
for ``P`` up to ``2^16``.

This is the splitting of Ozaki, Ogita, Oishi and Rump (2012).  It needs a hardware double precision product without
internal reduction of precision (no TF32 or bf16 accumulation); it does not repair rounding that already happened in
the derivative programs that produced ``J``, only the product that assembles ``K``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

Array = jax.Array

__all__ = ["exact_gram", "exact_matmul", "slice_bits", "split_slices"]


def slice_bits(inner: int, dtype_bits: int = 53) -> int:
    """Bits per slice so that a sum of ``inner`` slice products is exact: ``floor((bits - log2 inner) / 2)``."""
    if inner < 1:
        raise ValueError("the inner dimension must be positive")
    beta = int((dtype_bits - math.ceil(math.log2(max(inner, 2)))) // 2)
    if beta < 2:
        raise ValueError(
            f"inner dimension {inner} leaves fewer than 2 bits per slice at {dtype_bits} bits"
        )
    return beta


def split_slices(A: Array, *, axis: int, slices: int, beta: int) -> tuple[list[Array], Array]:
    """Split ``A`` into ``slices`` pieces of at most ``beta`` significant bits relative to the maximum along ``axis``.

    ``A = sum(pieces) + remainder`` exactly.  Each piece is an integer multiple of ``2**(e - beta s)`` (``e`` the exponent
    of the largest magnitude along ``axis``) with at most ``2**beta`` as that integer, so products of two pieces summed
    over ``axis`` stay below ``2**53`` and are evaluated without rounding.
    """
    mx = jnp.max(jnp.abs(A), axis=axis, keepdims=True)
    # frexp and ldexp are exact: log2 and exp2 are not, and a scale that is not a power of two breaks the exactness
    e = jnp.frexp(jnp.where(mx > 0, mx, 1.0))[1]  # mx <= 2**e
    one = jnp.ones_like(mx)
    rem = A
    pieces: list[Array] = []
    for s in range(1, slices + 1):
        scale = jnp.ldexp(one, e - beta * s)
        piece = jnp.round(rem / scale) * scale
        pieces.append(piece)
        rem = rem - piece
    return pieces, rem


def exact_matmul(A: Array, B: Array, *, slices: int = 4) -> Array:
    """``A @ B`` for float64 matrices, correct to a few ulps of the result regardless of the inner dimension.

    ``A`` is ``(m, P)``, ``B`` is ``(P, n)``.  The ``slices`` pieces carry ``slices * floor((53 - log2 P) / 2)`` bits of each
    operand; the part beyond that is dropped (a relative ``2**-(slices beta)`` of the row maximum).  With the default
    ``slices = 4`` a row whose entries span a factor ``2**w`` is represented to full float64 accuracy while
    ``4 beta >= 53 + w`` (``w`` up to about 30 for ``P`` up to ``2**16``); beyond that raise ``slices``.
    """
    if A.dtype != jnp.float64 or B.dtype != jnp.float64:
        raise TypeError("exact_matmul needs float64 operands")
    if A.shape[-1] != B.shape[0]:
        raise ValueError("inner dimensions do not match")
    beta = slice_bits(A.shape[-1])
    a, _ = split_slices(A, axis=1, slices=slices, beta=beta)
    b, _ = split_slices(B, axis=0, slices=slices, beta=beta)
    terms = [(s, t) for s in range(slices) for t in range(slices) if s + t < slices]
    # smallest terms first: the sum then rounds once per addition at the scale of the running total
    acc = jnp.zeros((A.shape[0], B.shape[1]), A.dtype)
    for s, t in sorted(terms, key=lambda st: -(st[0] + st[1])):
        acc = acc + a[s] @ b[t]
    return acc


def exact_gram(J: Array, *, slices: int = 4) -> Array:
    """``J @ J.T`` through :func:`exact_matmul`, symmetrised exactly (``K_ij`` and ``K_ji`` from the same sum)."""
    K = exact_matmul(J, J.T, slices=slices)
    return jnp.triu(K) + jnp.triu(K, 1).T
