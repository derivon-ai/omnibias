# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Pipage and swap rounding of a fractional matroid point to an integral independent set.

Both turn the continuous-greedy fractional ``p*`` into a feasible ``0/1`` set without
losing multilinear value:

* :func:`pipage_round` -- **deterministic**. Along a within-group transfer
  ``p_i += t, p_j -= t`` the multilinear extension of a submodular ``f`` is *convex*
  (its only second-order term is ``-2 d^2F/dp_i dp_j >= 0``), so moving to whichever
  endpoint has larger ``F`` never decreases it; iterating pins every coordinate to
  ``{0, 1}`` and yields ``f(S) >= F(p*)``.
* :func:`swap_round` -- **randomized** (seeded here for determinism). Merges the
  continuous-greedy basis sequence pairwise by matroid base exchanges (within a group,
  so capacities are preserved), with ``E[f(S)] >= F(p*)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from omnibias.submodular.functions import SubmodularFunction
from omnibias.submodular.matroid import Matroid

FloatArray = NDArray[np.float64]

_TOL = 1e-9


def _snap(p: FloatArray, i: int) -> None:
    if abs(p[i]) < _TOL:
        p[i] = 0.0
    elif abs(p[i] - 1.0) < _TOL:
        p[i] = 1.0


def _fractional(p: FloatArray, group: NDArray[np.int64]) -> list[int]:
    return [int(i) for i in group if _TOL < p[i] < 1.0 - _TOL]


def pipage_round(
    function: SubmodularFunction, matroid: Matroid, p: object
) -> tuple[tuple[int, ...], float]:
    r"""Deterministic pipage rounding; returns ``(selection, f(selection))``.

    Rounds within each matroid group (so capacities stay satisfied), always choosing the
    ``F``-nondecreasing endpoint, so the returned set satisfies ``f(S) >= F(p)``.
    """
    pv = np.asarray(p, dtype=float).reshape(-1).copy()
    for group, cap_raw in zip(matroid.groups(), matroid.caps(), strict=True):
        cap = min(int(cap_raw), int(group.size))
        frac = _fractional(pv, group)
        while len(frac) >= 2:
            i, j = frac[0], frac[1]
            a = min(1.0 - pv[i], pv[j])  # p_i += a, p_j -= a
            b = min(1.0 - pv[j], pv[i])  # p_j += b, p_i -= b
            up = pv.copy()
            up[i] += a
            up[j] -= a
            dn = pv.copy()
            dn[j] += b
            dn[i] -= b
            pv = up if float(function.multilinear(up)) >= float(function.multilinear(dn)) else dn
            _snap(pv, i)
            _snap(pv, j)
            frac = _fractional(pv, group)
        if len(frac) == 1:
            i = frac[0]
            up = pv.copy()
            up[i] = 1.0
            dn = pv.copy()
            dn[i] = 0.0
            # both endpoints keep the group within capacity (see module note); pick the
            # F-nondecreasing one, which for monotone f rounds the last fraction up.
            group_others = float(np.sum(pv[group])) - pv[i]
            up_feasible = group_others + 1.0 <= cap + _TOL
            if up_feasible and float(function.multilinear(up)) >= float(function.multilinear(dn)):
                pv = up
            else:
                pv = dn
            _snap(pv, i)
    selection = tuple(int(round(v)) for v in pv)
    return selection, float(function.value(np.asarray(selection, dtype=float)))


def _merge_bases(
    b1: FloatArray,
    b2: FloatArray,
    w1: float,
    w2: float,
    groups: list[NDArray[np.int64]],
    rng: np.random.Generator,
) -> FloatArray:
    b1 = b1.copy()
    b2 = b2.copy()
    while not np.array_equal(b1, b2):
        moved = False
        for group in groups:
            only1 = [int(i) for i in group if b1[i] == 1.0 and b2[i] == 0.0]
            only2 = [int(i) for i in group if b2[i] == 1.0 and b1[i] == 0.0]
            if only1 and only2:
                i, j = only1[0], only2[0]
                if rng.random() < w1 / (w1 + w2):  # move b2 toward b1
                    b2[i], b2[j] = 1.0, 0.0
                else:  # move b1 toward b2
                    b1[i], b1[j] = 0.0, 1.0
                moved = True
                break
        if not moved:  # per-group counts differ (under-filled bases): stop with a valid base
            break
    return b1


def swap_round(
    function: SubmodularFunction,
    matroid: Matroid,
    bases: list[FloatArray],
    *,
    seed: int = 0,
) -> tuple[tuple[int, ...], float]:
    r"""Randomized swap rounding of the continuous-greedy ``bases``; ``(selection, f)``.

    Merges the equal-weight basis sequence pairwise by within-group matroid exchanges
    (seeded for determinism), giving a single feasible base with ``E[f(S)] >= F(p*)``.
    """
    if not bases:
        raise ValueError("bases must be non-empty")
    rng = np.random.default_rng(seed)
    groups = matroid.groups()
    current = np.asarray(bases[0], dtype=float).copy()
    weight = 1.0
    for nxt in bases[1:]:
        current = _merge_bases(current, np.asarray(nxt, dtype=float), weight, 1.0, groups, rng)
        weight += 1.0
    selection = tuple(int(round(v)) for v in current)
    return selection, float(function.value(np.asarray(selection, dtype=float)))


__all__ = [
    "pipage_round",
    "swap_round",
]
