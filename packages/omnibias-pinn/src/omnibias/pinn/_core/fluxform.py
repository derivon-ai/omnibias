# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Index bookkeeping for the antisymmetric flux potential (backend-free).

The flux-form cage represents a divergence-free space-time vector as
``G^i = sum_j d_j A^{ij}`` with ``A^{ij} = -A^{ji}``. All that requires from
shared code is the *ordering* of the independent potential components and the
sign table that extends them to both index orders -- pure Python, so the torch
and jax cages cannot disagree about which name means which pair.
"""

from __future__ import annotations

from itertools import combinations


def antisymmetric_pairs(n_axes: int) -> tuple[tuple[int, int], ...]:
    """The ``(i, j)`` with ``i < j``, in the order potentials are supplied.

    ``D (D - 1) / 2`` pairs in lexicographic order: ``(0,1), (0,2), ..., (1,2),
    ...``. ``D = 2`` gives the single streamfunction pair and ``D = 3`` the
    three vector-potential pairs, so the conventional cages are the small cases
    of this ordering rather than exceptions to it.
    """
    if n_axes < 2:
        raise ValueError(f"need at least 2 axes for a flux form, got {n_axes}")
    return tuple(combinations(range(n_axes), 2))


def potential_table(
    n_axes: int, potential_names: tuple[str, ...]
) -> dict[tuple[int, int], tuple[str, float]]:
    """Map every ordered ``(i, j)``, ``i != j``, to ``(name, sign)``.

    Storing both orders with the sign folded in means the cage never branches
    on ``i < j`` at call time; ``A^{ii}`` is simply absent, which is the same
    statement as ``A^{ii} = 0``.
    """
    pairs = antisymmetric_pairs(n_axes)
    if len(potential_names) != len(pairs):
        raise ValueError(
            f"an antisymmetric potential on {n_axes} axes has {len(pairs)} "
            f"independent components {pairs!r}, but {len(potential_names)} "
            f"names were given: {potential_names!r}"
        )
    table: dict[tuple[int, int], tuple[str, float]] = {}
    for (i, j), name in zip(pairs, potential_names, strict=True):
        table[(i, j)] = (name, 1.0)
        table[(j, i)] = (name, -1.0)
    return table


__all__ = [
    "antisymmetric_pairs",
    "potential_table",
]
