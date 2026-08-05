# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-agnostic containers for omnibias-submodular.

:class:`SubmodularProblem` pairs a :class:`~omnibias.submodular.functions.SubmodularFunction`
with a :class:`~omnibias.submodular.matroid.Matroid` and implements the
``omnibias-discrete`` :class:`~omnibias.discrete.DiscreteProblem` seam via ``energy = -f``
(so *minimizing* the substrate energy *maximizes* ``f``) and ``to_polynomial = -F``. The
substrate's ``brute_force_min`` / ``certify_gap`` therefore apply to the (for monotone
``f``, trivial) *unconstrained* view; the headline constrained pipeline lives in
:mod:`omnibias.submodular` (continuous greedy + rounding + the matroid-constrained gap).

:class:`SubmodularSolution` is a decoded feasible set (a *lower* bound on ``OPT``);
:class:`SubmodularCertificate` sandwiches ``OPT`` between that decoded value and a
rigorous *upper* bound, so ``value <= OPT <= upper_bound`` is a certified gap -- never an
exact-optimality (``P = NP``) claim.

Terminology: the relaxation these containers feed hardens ``sigmoid(beta (g - tau))`` as
``beta -> inf`` -- the feasibility / temperature sense of "collapse", distinct from the
**founding bias collapse** (the multi-bias ``delta -> 0`` limit to the closed-form
derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from omnibias.submodular.functions import SubmodularFunction
from omnibias.submodular.matroid import Matroid

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.sos import Polynomial

FloatArray = NDArray[np.float64]

_TINY = 1e-12

# The classical continuous-greedy + rounding approximation ratio for monotone submodular
# maximization under a matroid constraint.
ONE_MINUS_INV_E = 1.0 - exp(-1.0)


@dataclass(frozen=True)
class ContinuousGreedySchedule:
    r"""Hyperparameters for the continuous-greedy (Frank-Wolfe) relaxation.

    Attributes
    ----------
    steps:
        Number of Frank-Wolfe steps ``T`` (each moves ``1/T`` toward a matroid basis).
        More steps shrink the ``O(1/T)`` discretization loss in ``F(p*) >= (1-1/e) OPT``.
    beta:
        Inverse temperature of the differentiable soft LP oracle (backend twins only);
        larger hardens the selection toward the exact basis. Ignored by the exact numpy
        path, which uses the hard :meth:`~omnibias.submodular.matroid.Matroid.max_weight_basis`.
    """

    steps: int = 25
    beta: float = 50.0

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.beta <= 0.0:
            raise ValueError("beta must be > 0")

    @classmethod
    def fast(cls) -> ContinuousGreedySchedule:
        """A softer, shorter schedule for training *through* the relaxation."""
        return cls(steps=15, beta=6.0)


@dataclass(frozen=True)
class SubmodularProblem:
    r"""Maximize a monotone submodular ``function`` over the ``matroid``'s independent sets.

    Implements the ``omnibias-discrete`` ``DiscreteProblem`` seam with ``energy = -f`` and
    ``to_polynomial = -F`` (the negated multilinear polynomial), so the whole substrate
    pipeline is available for the unconstrained view.
    """

    function: SubmodularFunction
    matroid: Matroid
    name: str | None = None

    def __post_init__(self) -> None:
        if self.function.n != self.matroid.n:
            raise ValueError(
                f"function/matroid ground-set mismatch: {self.function.n} vs {self.matroid.n}"
            )

    @property
    def n(self) -> int:
        return int(self.function.n)

    def energy(self, x: object) -> float | FloatArray:
        r"""The substrate energy ``E(x) = -f(x)`` (minimizing ``E`` maximizes ``f``)."""
        xv = np.asarray(x, dtype=float)
        v = self.function.value(xv)
        return -float(v) if xv.ndim == 1 else -np.asarray(v, dtype=float)

    def to_polynomial(self) -> Polynomial:
        """The energy ``-f`` as an :class:`omnibias.sos.Polynomial` (the negated ``F``)."""
        return -self.function.to_polynomial()


@dataclass(frozen=True)
class SubmodularSolution:
    r"""A decoded feasible set and (optionally) the fractional point it was rounded from.

    Attributes
    ----------
    selection:
        The ``0/1`` indicator of the chosen set (length ``n``); a *lower* bound witness
        (``f(selection) <= OPT``).
    value:
        ``f(selection)`` -- the achieved (lower-bound) objective.
    fractional:
        The continuous-greedy fractional point ``p* in [0, 1]^n`` (or ``None``).
    fractional_value:
        ``F(p*)`` -- the multilinear objective before rounding (or ``None``).
    rounding:
        Which rounding produced ``selection`` (``"pipage"`` / ``"swap"``), or ``None``.
    """

    selection: tuple[int, ...]
    value: float
    fractional: FloatArray | None = None
    fractional_value: float | None = None
    rounding: str | None = None

    @property
    def n(self) -> int:
        return len(self.selection)

    @property
    def support(self) -> tuple[int, ...]:
        """The indices of the chosen elements (``{i : selection_i = 1}``)."""
        return tuple(i for i, v in enumerate(self.selection) if v)


@dataclass(frozen=True)
class SubmodularCertificate:
    r"""A rigorous approximation / optimality-gap certificate for a decoded set.

    Sandwiches the true constrained optimum ``OPT`` between the decoded value (a valid
    *lower* bound, since the set is feasible) and a rigorous data-dependent *upper*
    bound, so ``value <= OPT <= upper_bound`` -- a certified gap, **never** an
    exact-optimality (``P = NP``) claim. :attr:`approx_ratio` is the a-priori
    ``1 - 1/e`` guarantee that continuous greedy + rounding carries for monotone ``f``.

    Attributes
    ----------
    value:
        ``f(S)`` for the decoded feasible set ``S`` (a lower bound on ``OPT``).
    upper_bound:
        A rigorous upper bound ``U(S) >= OPT`` (the marginal-gain / modular bound).
    fractional_value:
        ``F(p*)`` from continuous greedy, when available (``>= (1-1/e) OPT``).
    approx_ratio:
        The a-priori guarantee ``1 - 1/e`` (monotone submodular + matroid constraint).
    method:
        Which upper bound was used (``"marginal"``).
    curvature:
        The total curvature ``c in [0, 1]`` of ``f`` when requested (else ``None``); a
        smaller ``c`` sharpens the guarantee via :attr:`curvature_ratio`.
    """

    value: float
    upper_bound: float
    fractional_value: float | None
    approx_ratio: float
    method: str
    curvature: float | None = None

    @property
    def curvature_ratio(self) -> float:
        r"""The curvature-sharpened a-priori ratio ``(1/c)(1 - e^{-c}) >= 1 - 1/e``.

        With total curvature ``c``, continuous greedy / greedy carry the stronger a-priori
        guarantee ``(1/c)(1 - e^{-c})`` (which decreases from ``1`` at ``c = 0`` to ``1 - 1/e``
        at ``c = 1``), so this is never below :attr:`approx_ratio`. Falls back to
        :attr:`approx_ratio` when :attr:`curvature` was not computed.
        """
        if self.curvature is None:
            return self.approx_ratio
        c = self.curvature
        if c <= _TINY:
            return 1.0
        return float((1.0 / c) * (1.0 - exp(-c)))

    @property
    def absolute_gap(self) -> float:
        """Certified absolute gap ``upper_bound - value`` (``>= 0``)."""
        return self.upper_bound - self.value

    @property
    def relative_gap(self) -> float:
        """Certified relative gap ``(upper_bound - value) / max(|upper_bound|, tiny)``."""
        return self.absolute_gap / max(abs(self.upper_bound), _TINY)

    @property
    def certified_ratio(self) -> float:
        r"""A *sound* lower bound on the achieved ratio ``f(S) / OPT`` (``value/upper_bound``).

        Since ``OPT <= upper_bound``, ``f(S)/OPT >= value/upper_bound`` -- a certified
        (possibly looser than ``1 - 1/e``) runtime ratio; a weaker ``U`` only lowers it,
        never makes it unsound.
        """
        if self.upper_bound <= _TINY:
            return 1.0
        return float(self.value / self.upper_bound)

    @property
    def internal_consistent(self) -> bool:
        r"""Whether the reported pair is coherent: ``value <= upper_bound``.

        This is a cheap self-check, **not** a soundness claim. It compares the
        certificate against itself, so it stays ``True`` when *both* sides sit far
        below the true optimum -- exactly the state an invalid upper bound produces.
        It was previously named ``is_sound``, which invited it to be read as the
        guarantee ``OPT <= upper_bound``; that is a different statement, and the only
        way to check it directly is against an oracle.

        Use :func:`~omnibias.submodular.verify_guarantee` (exponential, small ``n``) for
        the real sandwich ``f(S) <= OPT <= U(S)``. Soundness of the bound itself is
        established by its derivation, which is why
        :func:`~omnibias.submodular.certify_submodular_gap` selects one whose hypotheses
        the problem actually satisfies.
        """
        return self.value <= self.upper_bound + 1e-9


__all__ = [
    "ContinuousGreedySchedule",
    "ONE_MINUS_INV_E",
    "SubmodularCertificate",
    "SubmodularProblem",
    "SubmodularSolution",
]
