# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differential-form bookkeeping (pure Python: no torch / jax).

A ``DifferentialForm`` is a symbolic ``k``-form whose components are *field
component names* (resolved against a :class:`FieldState` by the backend ops). The
combinatorial helpers here (sorted index sets, permutation signs, the wedge
product on already-evaluated component dicts) are backend-agnostic: the wedge
operates on dicts of tensors using only ``+`` / ``*``, which are duck-typed on
torch and jax tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any


def sorted_index_sets(dim: int, k: int) -> list[tuple[int, ...]]:
    """All strictly increasing ``k``-index tuples drawn from ``range(dim)``."""
    if k < 0 or k > dim:
        return []
    if k == 0:
        return [()]
    return [tuple(c) for c in combinations(range(dim), k)]


def permutation_sign(perm: tuple[int, ...]) -> int:
    """Sign (+/-1) of a permutation given as a sequence of distinct ints."""
    seq = list(perm)
    sign = 1
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                sign = -sign
    return sign


@dataclass(frozen=True)
class DifferentialForm:
    """A symbolic ``k``-form over a field.

    Parameters
    ----------
    degree
        The form degree ``k``.
    dim
        The manifold dimension ``d``.
    comps
        Mapping from a strictly increasing ``k``-index tuple to the field
        component name holding that component's value. Missing tuples are zero.
    """

    degree: int
    dim: int
    comps: dict[tuple[int, ...], str]

    def __post_init__(self) -> None:
        for idx in self.comps:
            if len(idx) != self.degree:
                raise ValueError(
                    f"index {idx!r} does not match degree {self.degree}"
                )
            if list(idx) != sorted(idx) or len(set(idx)) != len(idx):
                raise ValueError(f"index {idx!r} must be strictly increasing")


def wedge(
    a: dict[tuple[int, ...], Any],
    ka: int,
    b: dict[tuple[int, ...], Any],
    kb: int,
    dim: int,
) -> dict[tuple[int, ...], Any]:
    r"""Wedge product of two evaluated forms.

    ``a`` and ``b`` map sorted index tuples to component *tensors* (shape
    ``(B,)``). Returns the ``(ka + kb)``-form

    .. math::

        (\alpha\wedge\beta)_{I} = \sum_{\substack{A\sqcup B = I}}
            \mathrm{sgn}(A, B)\,\alpha_A\,\beta_B,

    summed over partitions of each sorted output index set ``I`` into an
    ``a``-part ``A`` and a ``b``-part ``B``.
    """
    out: dict[tuple[int, ...], Any] = {}
    for ia, va in a.items():
        for ib, vb in b.items():
            if set(ia) & set(ib):
                continue  # repeated index -> wedge vanishes
            merged = ia + ib
            target = tuple(sorted(merged))
            # sign of the permutation taking (merged) to (sorted target)
            order = sorted(range(len(merged)), key=lambda p: merged[p])
            sign = permutation_sign(tuple(order))
            contrib = (sign * va) * vb if sign != 1 else va * vb
            if target in out:
                out[target] = out[target] + contrib
            else:
                out[target] = contrib
    return out


def interior_product(
    vector: Any,
    values: dict[tuple[int, ...], Any],
    degree: int,
    dim: int,
) -> dict[tuple[int, ...], Any]:
    r"""Interior product (contraction) :math:`\iota_X\omega` of a vector with a form.

    ``vector`` is a tensor of shape ``(B, dim)`` (the contravariant components
    :math:`X^j`); ``values`` is an evaluated ``degree``-form (sorted-index dict of
    ``(B,)`` tensors). Returns the evaluated ``(degree-1)``-form

    .. math::

        (\iota_X\omega)_{j_1\dots j_{k-1}}
            = \sum_{m} X^m\,\omega_{m j_1\dots j_{k-1}}.

    A contraction of a 0-form is zero (empty dict). ``iota_X`` is a graded
    anti-derivation with :math:`\iota_X\iota_X = 0`.
    """
    if degree <= 0:
        return {}
    out: dict[tuple[int, ...], Any] = {}
    for j_set in sorted_index_sets(dim, degree - 1):
        acc: Any = None
        for m in range(dim):
            if m in j_set:
                continue
            merged = (m,) + j_set
            target = tuple(sorted(merged))
            val = values.get(target)
            if val is None:
                continue
            order = sorted(range(len(merged)), key=lambda p: merged[p])
            sign = permutation_sign(tuple(order))
            xm = vector[..., m]
            term = (sign * xm) * val if sign != 1 else xm * val
            acc = term if acc is None else acc + term
        if acc is not None:
            out[j_set] = acc
    return out


__all__ = [
    "DifferentialForm",
    "interior_product",
    "permutation_sign",
    "sorted_index_sets",
    "wedge",
]
