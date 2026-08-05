# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The :class:`DiscreteProblem` seam and the Boolean-hypercube constraint generators.

Any object that implements :class:`DiscreteProblem` plugs into the whole substrate --
the annealed relaxation (given an energy gradient), the rounding / local-search decoder,
the brute-force oracle, and the SOS / Lasserre certified lower bound. Consumers such as
``omnibias.qubo.QUBOProblem`` and ``omnibias.discrete.maxsat.MaxSATProblem`` specialise
the energy; the substrate is written once against this protocol.

:func:`boolean_constraints` builds the semialgebraic description of ``{0, 1}^n`` that
Putinar's Positivstellensatz uses -- the two inequalities ``x_i - x_i^2 >= 0`` and
``x_i^2 - x_i >= 0`` per variable, which together pin ``x_i(1 - x_i) = 0``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.sos import Polynomial

FloatArray = NDArray[np.float64]


@runtime_checkable
class DiscreteProblem(Protocol):
    r"""A pseudo-Boolean minimization problem over ``x in {0, 1}^n``.

    Required surface
    ----------------
    n:
        The number of binary variables.
    energy(x):
        The objective at one point ``(n,)`` or a batch ``(m, n)``; returns a ``float``
        for a single point and an ``(m,)`` array for a batch. Minimizing ``energy`` is
        the problem.
    to_polynomial():
        The energy as an :class:`omnibias.sos.Polynomial` over ``n`` variables, used by
        the certified Lasserre lower bound.

    Optional fast path
    ------------------
    flip_deltas(x):
        The vector of energy changes ``E(x with bit i flipped) - E(x)`` for every ``i``.
        When present the local-search decoder uses it instead of the generic
        batched-energy fallback (e.g. QUBO has a one-matvec closed form).
    """

    @property
    def n(self) -> int: ...

    def energy(self, x: object) -> float | FloatArray: ...

    def to_polynomial(self) -> Polynomial: ...


def boolean_constraints(n: int) -> list[Polynomial]:
    r"""The Boolean-hypercube generators ``x_i - x_i^2 >= 0`` and ``x_i^2 - x_i >= 0``.

    Together the two inequalities per variable pin ``x_i(1 - x_i) = 0``, i.e.
    ``x_i in {0, 1}`` -- the semialgebraic description Putinar's Positivstellensatz uses
    to certify a lower bound on the minimum energy over the cube.
    """
    from omnibias.sos import Polynomial

    cons: list[Polynomial] = []
    for i in range(n):
        xi = Polynomial.variable(i, n)
        xi_sq = xi * xi
        cons.append(xi - xi_sq)
        cons.append(xi_sq - xi)
    return cons


__all__ = ["DiscreteProblem", "boolean_constraints"]
