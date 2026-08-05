# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified upper bound on the constrained optimum (the gap certificate's upper side).

For a monotone submodular ``f``, any feasible set ``S`` and the optimal independent set
``O``, submodularity gives ``f(O) <= f(S) + sum_{i in O\S} [f(S + i) - f(S)]``. Since
``O\S`` is itself independent and the marginal gains are nonnegative,

.. math::
    OPT = f(O) \le f(S) + \max_{I\ \text{independent}} \sum_{i\in I} [f(S+i)-f(S)]_+,

and the inner maximum is exactly the matroid linear oracle on the (clamped) marginal
gains. This yields the **sound** upper bound :func:`marginal_upper_bound` -- a weaker
bound only widens the certified gap, it is never unsound.

A second **sound** upper bound comes from the *modular* over-approximation of ``f``: by
submodularity every marginal is dominated by the corresponding singleton gain, so
``f(X) <= f(empty) + sum_{i in X} [f({i}) - f(empty)]`` and hence
``OPT <= f(empty) + max_{I independent} sum_{i in I} [f({i}) - f(empty)]_+``
(:func:`modular_upper_bound`). It does not depend on the decoded ``S``; taking the ``min`` of
it and :func:`marginal_upper_bound` is still sound and only ever *tightens* the certified gap.

The substrate's Lasserre / SOS lower bound (:func:`lasserre_lower_bound`) and its trivial
back-stop (:func:`negative_coeff_lower_bound`) are re-exported for the *unconstrained*
SOS certificate on the multilinear polynomial (``omnibias-discrete`` substrate).

:func:`total_curvature` measures how far ``f`` is from modular; a small curvature ``c``
sharpens the a-priori guarantee from ``1 - 1/e`` to ``(1/c)(1 - e^{-c})`` (see
:attr:`~omnibias.submodular.problem.SubmodularCertificate.curvature_ratio`).
"""

from __future__ import annotations

import numpy as np
from omnibias.discrete import lasserre_lower_bound, negative_coeff_lower_bound
from omnibias.submodular.functions import SubmodularFunction
from omnibias.submodular.matroid import Matroid

_TOL = 1e-12


def marginal_upper_bound(function: SubmodularFunction, matroid: Matroid, x: object) -> float:
    r"""A rigorous upper bound ``U(S) >= OPT`` from the marginal gains at ``S``.

    ``U(S) = f(S) + max_{I independent} sum_{i in I} [f(S + i) - f(S)]_+``.

    **Monotone ``f`` only**, and the requirement is enforced. Submodularity alone gives
    ``f(O u S) <= f(S) + sum_{i in O\S} [f(S+i) - f(S)]``; getting from there to the
    optimum needs ``f(O) <= f(O u S)``, which is exactly monotonicity. Drop it and the
    bound can land strictly *below* ``OPT`` -- on random :class:`~omnibias.submodular.GraphCut`
    instances it does so essentially always, and the resulting certificate looks perfectly
    self-consistent (``f(S) <= U``) while asserting a bound that is false. Non-monotone
    problems belong on :func:`modular_upper_bound` /
    :func:`~omnibias.submodular.nonmonotone_upper_bound`, which
    :func:`~omnibias.submodular.certify_submodular_gap` selects automatically.
    """
    if not function.is_monotone:
        raise ValueError(
            "marginal_upper_bound requires a monotone f: the step f(O) <= f(O u S) does "
            "not hold otherwise, and the result can fall below the true optimum. Use "
            "modular_upper_bound / nonmonotone_upper_bound, or call "
            "certify_submodular_gap, which selects a valid bound from function.is_monotone."
        )
    xv = np.asarray(x, dtype=float).reshape(-1)
    f_s = float(function.value(xv))
    marginals = np.maximum(function.marginal_gains(xv), 0.0)
    basis = matroid.max_weight_basis(marginals)
    return f_s + float(np.sum(marginals * basis))


def modular_upper_bound(function: SubmodularFunction, matroid: Matroid) -> float:
    r"""A rigorous ``S``-independent upper bound from the modular over-approximation of ``f``.

    ``U_mod = f(empty) + max_{I independent} sum_{i in I} [f({i}) - f(empty)]_+``.

    Sound for **any** submodular ``f``, monotone or not: telescoping ``f(X) - f(empty)``
    and dominating each marginal by its singleton gain uses only diminishing returns, so
    ``m(X) = f(empty) + sum_{i in X} [f({i}) - f(empty)]`` upper-bounds ``f`` everywhere
    and its constrained maximum is the matroid linear oracle on the singleton gains. That
    is why this, and not :func:`marginal_upper_bound`, carries the non-monotone path.

    Complements :func:`marginal_upper_bound` (which is evaluated *at* ``S``); the two are
    incomparable in general, so :func:`~omnibias.submodular.certify_submodular_gap` takes
    the tighter ``min`` whenever both are available.
    """
    n = function.n
    zero = np.zeros(n, dtype=float)
    f_empty = float(function.value(zero))
    singles = np.maximum(function.marginal_gains(zero), 0.0)  # [f({i}) - f(empty)]_+
    basis = matroid.max_weight_basis(singles)
    return f_empty + float(np.sum(singles * basis))


def total_curvature(function: SubmodularFunction, ground: object | None = None) -> float:
    r"""The total curvature ``c = 1 - min_i [f(V) - f(V \ i)] / [f({i}) - f(empty)]`` in ``[0, 1]``.

    Curvature measures how far a monotone submodular ``f`` is from *modular*: ``c = 0`` iff
    ``f`` is modular (marginals never shrink), ``c -> 1`` for a fully saturating ``f``. The
    minimum ranges over elements of the ground set ``V`` with a strictly positive standalone
    gain (an element with ``f({i}) = f(empty)`` cannot bind the curvature and is skipped).

    ``ground`` is an optional ``0/1`` indicator (or index iterable) selecting ``V``; it
    defaults to the full set ``[n]``. The result is clamped to ``[0, 1]`` against rounding.

    Monotone ``f`` only: the ratio it minimises presumes nonnegative marginals, and the
    guarantee it sharpens is itself a monotone-only theorem.
    """
    if not function.is_monotone:
        raise ValueError(
            "total_curvature is defined for monotone f only: it sharpens the (1 - 1/e) "
            "guarantee, which does not apply to a non-monotone function."
        )
    n = function.n
    if ground is None:
        v = np.ones(n, dtype=float)
    else:
        gv = np.asarray(ground, dtype=float).reshape(-1)
        if gv.shape[0] == n and np.all((gv == 0.0) | (gv == 1.0)):
            v = gv.copy()
        else:  # treat as an iterable of indices
            v = np.zeros(n, dtype=float)
            for i in np.asarray(ground, dtype=np.int64).reshape(-1):
                v[int(i)] = 1.0
    f_empty = float(function.value(np.zeros(n, dtype=float)))
    f_v = float(function.value(v))
    min_ratio = 1.0
    for i in np.where(v > 0.5)[0]:
        single = np.zeros(n, dtype=float)
        single[i] = 1.0
        denom = float(function.value(single)) - f_empty  # f({i}) - f(empty)
        if denom <= _TOL:
            continue  # no standalone value -> does not constrain the curvature
        v_minus = v.copy()
        v_minus[i] = 0.0
        num = f_v - float(function.value(v_minus))  # f(V) - f(V \ i)
        min_ratio = min(min_ratio, num / denom)
    return float(min(max(1.0 - min_ratio, 0.0), 1.0))


__all__ = [
    "lasserre_lower_bound",
    "marginal_upper_bound",
    "modular_upper_bound",
    "negative_coeff_lower_bound",
    "total_curvature",
]
