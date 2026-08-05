# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The certified approximation / optimality-gap sandwich for a decoded feasible set.

:func:`certify_submodular_gap` sandwiches the constrained optimum ``OPT`` between

* the decoded value ``f(S)`` (a *lower* bound, since ``S`` is feasible) -- and
* a rigorous upper bound ``U(S) >= OPT``.

**Which** upper bound is not a free choice. The marginal-gain bound
(:func:`~omnibias.submodular._core.bound.marginal_upper_bound`) is derived under
monotonicity and is invalid without it, so this module reads
:attr:`~omnibias.submodular.functions.SubmodularFunction.is_monotone` and takes the
tightest bound whose hypotheses hold: the marginal bound for monotone ``f``, and the
modular / singleton bounds (:func:`~omnibias.submodular._core.bound.modular_upper_bound`,
:func:`~omnibias.submodular._core.nonmonotone.nonmonotone_upper_bound`, both derived
from diminishing returns alone) otherwise.

The result is a certified gap ``f(S) <= OPT <= U(S)``, plus the a-priori ``1 - 1/e``
guarantee **when ``f`` is monotone** -- never an exact-optimality (``P = NP``) claim.
:func:`verify_guarantee` self-checks the sandwich, and the a-priori ratio the
certificate actually claims, against the exact
:func:`~omnibias.submodular._core.greedy.brute_force_max` on small ``n``.

:func:`certify_unconstrained_gap` is a thin passthrough to the ``omnibias-discrete``
:func:`~omnibias.discrete.certify_gap` for the secondary *unconstrained* SOS bound on the
multilinear polynomial (for monotone ``f`` the unconstrained optimum is the full set, so
this is a soundness demonstration of the shared substrate seam, not the headline).

:func:`certify_nonmonotone_gap` is the analogue for a **non-monotone** ``f`` (e.g. a graph
cut): the marginal bound needs monotonicity, so it instead uses the sound singleton bound
:func:`~omnibias.submodular._core.nonmonotone.nonmonotone_upper_bound` and records the
a-priori ``1/2`` (unconstrained double greedy) or ``1/e`` (matroid measured continuous
greedy) ratio -- again a certified gap, never an exactness claim.
"""

from __future__ import annotations

from math import exp
from typing import TYPE_CHECKING

import numpy as np
from omnibias.submodular._core.bound import (
    marginal_upper_bound,
    modular_upper_bound,
    total_curvature,
)
from omnibias.submodular._core.greedy import brute_force_max
from omnibias.submodular._core.nonmonotone import nonmonotone_upper_bound
from omnibias.submodular.functions import SubmodularFunction
from omnibias.submodular.matroid import Matroid
from omnibias.submodular.problem import (
    ONE_MINUS_INV_E,
    SubmodularCertificate,
    SubmodularProblem,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.discrete import GapCertificate

# The non-monotone a-priori ratios: double greedy (unconstrained) vs measured continuous
# greedy (matroid).
_INV_E = exp(-1.0)
_DOUBLE_GREEDY_RATIO = 0.5


def _check_feasible_binary(problem: SubmodularProblem, selection: object) -> np.ndarray:
    xv = np.asarray(selection, dtype=float).reshape(-1)
    if xv.shape[0] != problem.n:
        raise ValueError(f"selection must have length {problem.n}, got {xv.shape[0]}")
    if not np.all((xv == 0.0) | (xv == 1.0)):
        raise ValueError("selection must be a 0/1 indicator (round the relaxation first)")
    if not problem.matroid.is_independent(xv):
        raise ValueError("selection must be feasible (independent in the matroid)")
    return xv


def certify_submodular_gap(
    problem: SubmodularProblem,
    selection: object,
    *,
    fractional: object | None = None,
    with_curvature: bool = False,
) -> SubmodularCertificate:
    r"""Certify how close the feasible set ``selection`` is to the constrained optimum.

    Parameters
    ----------
    problem:
        The monotone submodular maximization instance.
    selection:
        A feasible ``0/1`` indicator (e.g. ``maximize(...).selection``) -- the lower
        bound ``f(S) <= OPT``.
    fractional:
        The continuous-greedy fractional point ``p*`` (optional); records ``F(p*)``.
    with_curvature:
        If ``True``, additionally compute the total curvature ``c`` of ``f`` (an extra
        ``O(n)`` value evaluations) and store it, sharpening the a-priori guarantee to
        :attr:`~omnibias.submodular.problem.SubmodularCertificate.curvature_ratio`.

    Returns
    -------
    :class:`~omnibias.submodular.problem.SubmodularCertificate` with the decoded value,
    the rigorous upper bound ``U(S)``, and the a-priori ``approx_ratio = 1 - 1/e``.
    """
    xv = _check_feasible_binary(problem, selection)
    value = float(problem.function.value(xv))
    monotone = problem.function.is_monotone
    # Pick from bounds whose derivation actually applies. The marginal bound needs
    # monotonicity (see `marginal_upper_bound`); the modular / singleton bounds do not.
    # Taking the min of *valid* bounds stays valid and only ever tightens the gap --
    # but a min that includes an inapplicable bound is how a gap certificate ends up
    # asserting something false, so the non-monotone branch simply omits it.
    candidates = [modular_upper_bound(problem.function, problem.matroid)]
    if monotone:
        candidates.append(marginal_upper_bound(problem.function, problem.matroid, xv))
    else:
        candidates.append(nonmonotone_upper_bound(problem.function))
    upper = min(candidates)
    fractional_value = (
        None if fractional is None else float(problem.function.multilinear(fractional))
    )
    curvature = total_curvature(problem.function) if with_curvature else None
    return SubmodularCertificate(
        value=value,
        upper_bound=upper,
        fractional_value=fractional_value,
        # The (1 - 1/e) greedy guarantee is a monotone-only theorem. On a non-monotone f
        # the monotone maximizers carry no a-priori ratio at all, so claim none; the
        # measured/double-greedy ratios belong to `certify_nonmonotone_gap`, which knows
        # which maximizer produced the selection.
        approx_ratio=ONE_MINUS_INV_E if monotone else 0.0,
        method="marginal" if monotone else "modular_nonmonotone",
        curvature=curvature,
    )


def verify_guarantee(
    problem: SubmodularProblem, selection: object, *, max_n: int = 20, tol: float = 1e-9
) -> bool:
    r"""Self-check the sandwich and the ``(1 - 1/e)`` guarantee vs the exact oracle.

    Runs :func:`~omnibias.submodular._core.greedy.brute_force_max` (exponential, small
    ``n``) and returns ``True`` iff ``f(S) <= OPT <= U(S)`` and
    ``f(S) >= (1 - 1/e) OPT`` (up to ``tol``).
    """
    xv = _check_feasible_binary(problem, selection)
    cert = certify_submodular_gap(problem, xv)
    _, opt = brute_force_max(problem.function, problem.matroid, max_n=max_n)
    sandwich = (cert.value <= opt + tol) and (opt <= cert.upper_bound + tol)
    # `approx_ratio` is 0.0 on the non-monotone path, where no a-priori guarantee is
    # claimed, so this degrades to the sandwich alone rather than testing a theorem
    # that does not apply.
    guarantee = cert.value >= cert.approx_ratio * opt - tol
    return bool(sandwich and guarantee)


def certify_nonmonotone_gap(
    function: SubmodularFunction,
    selection: object,
    *,
    matroid: Matroid | None = None,
) -> SubmodularCertificate:
    r"""Certify a feasible set for a **non-monotone** submodular ``function``.

    Parameters
    ----------
    function:
        The (possibly non-monotone) submodular objective, e.g.
        :class:`~omnibias.submodular.functions.GraphCut`.
    selection:
        A ``0/1`` indicator; when ``matroid`` is given it must be independent.
    matroid:
        The matroid constraint, or ``None`` for the unconstrained problem. Its presence
        selects the recorded a-priori ratio: ``1/e`` (matroid, measured continuous greedy)
        vs ``1/2`` (unconstrained, randomized double greedy).

    Returns
    -------
    :class:`~omnibias.submodular.problem.SubmodularCertificate` with the decoded value, the
    sound singleton upper bound :func:`~omnibias.submodular._core.nonmonotone.nonmonotone_upper_bound`,
    and ``method="nonmonotone-singleton"``.
    """
    xv = np.asarray(selection, dtype=float).reshape(-1)
    if xv.shape[0] != function.n:
        raise ValueError(f"selection must have length {function.n}, got {xv.shape[0]}")
    if not np.all((xv == 0.0) | (xv == 1.0)):
        raise ValueError("selection must be a 0/1 indicator (round the relaxation first)")
    if matroid is not None and not matroid.is_independent(xv):
        raise ValueError("selection must be feasible (independent in the matroid)")
    value = float(function.value(xv))
    upper = nonmonotone_upper_bound(function)
    ratio = _DOUBLE_GREEDY_RATIO if matroid is None else _INV_E
    return SubmodularCertificate(
        value=value,
        upper_bound=upper,
        fractional_value=None,
        approx_ratio=ratio,
        method="nonmonotone-singleton",
    )


def certify_unconstrained_gap(
    problem: SubmodularProblem, x: object, **kwargs: object
) -> GapCertificate:
    r"""Passthrough to the ``omnibias-discrete`` SOS certificate on ``energy = -f``.

    Certifies the (for monotone ``f``, trivial) *unconstrained* view via the shared
    substrate; the headline constrained certificate is :func:`certify_submodular_gap`.
    """
    from omnibias.discrete import certify_gap

    return certify_gap(problem, x, **kwargs)


__all__ = [
    "certify_nonmonotone_gap",
    "certify_submodular_gap",
    "certify_unconstrained_gap",
    "verify_guarantee",
]
