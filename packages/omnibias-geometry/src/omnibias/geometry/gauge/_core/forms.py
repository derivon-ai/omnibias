# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Lie-algebra-valued differential forms (pure Python: no torch / jax).

This generalizes the (scalar-valued) ``DifferentialForm`` of omnibias-geometry
with an *adjoint* index ``a = 0, ..., dim(g) - 1``. A
:class:`LieAlgebraValuedForm` of degree ``k`` maps each strictly increasing
``k``-index spacetime tuple to a *tuple of field component names*, one per Lie
algebra generator. A gauge connection ``A`` is such a form of degree 1; its
curvature / field strength ``F`` is a degree-2 form.

The small combinatorial helpers (sorted index sets, permutation signs, the
Levi-Civita symbol) are kept here so this package depends only on
``omnibias-fields`` (and numpy), not on ``omnibias-geometry``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


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


def levi_civita_symbol(dim: int) -> FloatArray:
    r"""The totally antisymmetric Levi-Civita symbol ``eps`` of rank ``dim``.

    ``eps[i_0, ..., i_{dim-1}]`` is the sign of the permutation
    ``(i_0, ..., i_{dim-1})`` of ``(0, ..., dim-1)``, and zero on repeats.
    """
    eps = np.zeros((dim,) * dim, dtype=np.float64)
    for perm in permutations(range(dim)):
        eps[perm] = float(permutation_sign(perm))
    return eps


@dataclass(frozen=True)
class LieAlgebraValuedForm:
    r"""A symbolic Lie-algebra-valued ``k``-form over a :class:`FieldState`.

    Parameters
    ----------
    degree
        The form degree ``k`` (spacetime antisymmetric indices).
    dim
        The spacetime dimension ``d`` (4 for Yang-Mills on ``R^4``).
    adjoint_dim
        The Lie algebra dimension ``dim(g)`` (the number of generators).
    comps
        Mapping from a strictly increasing ``k``-index spacetime tuple to a tuple
        of ``adjoint_dim`` :class:`FieldState` component names -- the value of
        that spacetime component for each generator ``T^a``. Missing tuples are
        zero.
    """

    degree: int
    dim: int
    adjoint_dim: int
    comps: dict[tuple[int, ...], tuple[str, ...]]

    def __post_init__(self) -> None:
        if self.degree < 0:
            raise ValueError(f"degree must be >= 0, got {self.degree}")
        for idx, names in self.comps.items():
            if len(idx) != self.degree:
                raise ValueError(f"index {idx!r} does not match degree {self.degree}")
            if list(idx) != sorted(idx) or len(set(idx)) != len(idx):
                raise ValueError(f"index {idx!r} must be strictly increasing")
            if any(v < 0 or v >= self.dim for v in idx):
                raise ValueError(f"index {idx!r} out of range for dim {self.dim}")
            if len(names) != self.adjoint_dim:
                raise ValueError(
                    f"component {idx!r} has {len(names)} names, "
                    f"expected adjoint_dim {self.adjoint_dim}"
                )


def wedge(
    a: dict[tuple[int, ...], Any],
    ka: int,
    b: dict[tuple[int, ...], Any],
    kb: int,
    dim: int,
) -> dict[tuple[int, ...], Any]:
    r"""Wedge product of two evaluated scalar-valued forms.

    ``a`` and ``b`` map sorted index tuples to component tensors. Returns the
    ``(ka + kb)``-form via the antisymmetrized tensor product. Operates only with
    ``+`` / ``*`` so it is duck-typed on torch and jax tensors.
    """
    out: dict[tuple[int, ...], Any] = {}
    for ia, va in a.items():
        for ib, vb in b.items():
            if set(ia) & set(ib):
                continue
            merged = ia + ib
            target = tuple(sorted(merged))
            order = sorted(range(len(merged)), key=lambda p: merged[p])
            sign = permutation_sign(tuple(order))
            contrib = (sign * va) * vb if sign != 1 else va * vb
            out[target] = contrib if target not in out else out[target] + contrib
    return out


def hodge_star_flat(
    values: dict[tuple[int, ...], Any],
    degree: int,
    dim: int,
    signature: tuple[int, ...],
) -> dict[tuple[int, ...], Any]:
    r"""Flat, signature-aware Hodge star of an evaluated ``k``-form.

    For the flat diagonal metric with entries ``signature`` (each ``+/-1``),
    ``|det g| = 1`` and the inverse-metric diagonal equals ``signature`` itself,
    so

    .. math::

        (\star\alpha)_J = \varepsilon(I, J)\,\Big(\prod_{\mu\in I}\eta_\mu\Big)\,
            \alpha_I,\qquad I = \{0,\dots,d-1\}\setminus J.

    Consequently ``** = (-1)^{k(d-k)} sign(det g)``, the standard identity that
    omnibias-geometry's metric Hodge star cannot express because it ignores the
    metric signature. Works component-wise, so each value may itself carry a
    trailing adjoint index.
    """
    if len(signature) != dim:
        raise ValueError(f"signature length {len(signature)} != dim {dim}")
    out: dict[tuple[int, ...], Any] = {}
    for j in sorted_index_sets(dim, dim - degree):
        i_comp = tuple(sorted(set(range(dim)) - set(j)))
        sign = permutation_sign(i_comp + j)
        factor = sign
        for mu in i_comp:
            factor *= signature[mu]
        am = values.get(i_comp)
        if am is None:
            continue
        out[j] = factor * am
    return out


__all__ = [
    "LieAlgebraValuedForm",
    "hodge_star_flat",
    "levi_civita_symbol",
    "permutation_sign",
    "sorted_index_sets",
    "wedge",
]
