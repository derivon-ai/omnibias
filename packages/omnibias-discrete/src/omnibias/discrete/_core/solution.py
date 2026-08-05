# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-agnostic result containers for the discrete substrate.

:class:`DiscreteSolution` is a decoded binary point (an *upper* bound on the minimum
energy); :class:`GapCertificate` sandwiches the true optimum between a rigorous *lower*
bound and that decoded energy, so ``lower_bound <= optimum <= energy`` is a **certified
gap** -- never an exact-optimality (P = NP) claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

ArrayT = TypeVar("ArrayT")

_TINY = 1e-12


@dataclass(frozen=True)
class DiscreteSolution(Generic[ArrayT]):
    r"""A decoded binary point and (optionally) the relaxation it was rounded from.

    Attributes
    ----------
    assignment:
        The decoded binary vector as a tuple of ``0`` / ``1`` of length ``n``.
    energy:
        The energy of ``assignment`` (an *upper* bound on the minimum).
    relaxed:
        Optional soft assignment ``x in (0, 1)^n`` produced by the differentiable
        relaxation (the point the binary vector was decoded from), a backend array.
    """

    assignment: tuple[int, ...]
    energy: float
    relaxed: ArrayT | None = None

    @property
    def n(self) -> int:
        return len(self.assignment)


@dataclass(frozen=True)
class GapCertificate:
    r"""A rigorous optimality-gap certificate for a decoded discrete point.

    Combines a rigorous **lower** bound on the minimum energy with the decoded point's
    energy as the **upper** bound, so the true optimum is provably sandwiched
    ``lower_bound <= optimum <= energy``; the gap certifies how close to optimal the
    point is -- **without** any exact-optimality (P = NP) claim.

    Attributes
    ----------
    lower_bound:
        Rigorous lower bound on the minimum energy of the problem.
    energy:
        The decoded point's energy (the certified upper bound).
    method:
        Which lower bound was used, e.g. ``"sos"`` (Lasserre / Positivstellensatz over
        the Boolean hypercube), ``"spectral"`` (eigenvalue-shift / box-QP relaxation),
        ``"negative_coeff"`` (the trivial sum-of-negative-coefficients bound), or
        ``"none"`` (no lower bound available in this environment).
    level:
        The SOS relaxation half-degree for ``method == "sos"`` (``0`` otherwise).
    certified:
        ``True`` iff ``lower_bound`` is rigorously sealed (an SOS proof, or an
        interval-sealed convex bound); ``False`` for a valid-but-unsealed float value.
    sealed:
        Optional hash-sealed v1 certificate (an ``omnibias.core.proof`` object) when an
        SOS lower bound was proved and sealed; ``None`` otherwise.
    """

    lower_bound: float
    energy: float
    method: str
    level: int
    certified: bool
    sealed: object | None = None

    @property
    def absolute_gap(self) -> float:
        """Certified absolute optimality gap ``energy - lower_bound`` (``>= 0``)."""
        return self.energy - self.lower_bound

    @property
    def relative_gap(self) -> float:
        """Certified relative gap ``(energy - lower_bound) / max(|lower_bound|, tiny)``."""
        return self.absolute_gap / max(abs(self.lower_bound), _TINY)

    @property
    def is_sound(self) -> bool:
        """Whether the sandwich holds (``lower_bound <= energy`` within rounding)."""
        return self.lower_bound <= self.energy + 1e-9


__all__ = ["DiscreteSolution", "GapCertificate"]
