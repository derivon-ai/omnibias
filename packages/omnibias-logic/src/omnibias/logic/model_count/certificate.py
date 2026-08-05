# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""A rigorous count-enclosure certificate for (weighted) #SAT / model counting.

:class:`CountCertificate` sandwiches the true (weighted) number of satisfying assignments
between a rigorous **lower** and **upper** bound, ``lower <= #models <= upper`` -- an
*enclosure*, never a poly-time exact-count (``#P`` / ``P = NP``) claim. Exact (weighted)
model counting is ``#P``-hard, so the honest deliverable is a certified sandwich whose
width a caller can only *shrink* by paying for more work (a higher inclusion-exclusion
order), never fake to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.core.proof.certificate import Cert
    from omnibias.logic.model_count.problem import ModelCountProblem

_TINY = 1e-12


@dataclass(frozen=True)
class CountCertificate:
    r"""A rigorous ``lower <= #models <= upper`` enclosure of the (weighted) model count.

    Attributes
    ----------
    lower:
        Rigorous lower bound on the (weighted) number of satisfying assignments.
    upper:
        Rigorous upper bound on the same count (``upper >= lower``).
    method:
        Which enclosure produced the bounds, e.g. ``"inclusion_exclusion"`` (truncated
        Bonferroni over the clause-violation events) or ``"trivial"`` (the ``[0, Z0]``
        floor when no terms were summed).
    order:
        The Bonferroni truncation order used (number of inclusion-exclusion terms summed);
        a higher order only tightens the enclosure. ``order >= #clauses`` is exact.
    weighted:
        ``True`` iff per-variable literal weights were supplied (weighted model counting),
        ``False`` for plain (unweighted) ``#SAT``.
    tight:
        ``True`` iff the enclosure collapsed to a single value (``lower == upper`` in exact
        arithmetic) -- i.e. the truncation order was high enough to count exactly for *this*
        instance. Instance-dependent; full inclusion-exclusion is exponential in the number
        of clauses, so this is a report, not a poly-time exactness guarantee.
    total:
        The total measure ``Z0`` of the assignment space (``2^n`` unweighted, or the product
        of per-variable weight sums when weighted); ``#models`` and both bounds lie in
        ``[0, Z0]``.
    """

    lower: float
    upper: float
    method: str
    order: int
    weighted: bool
    tight: bool
    total: float

    @property
    def width(self) -> float:
        """Certified enclosure width ``upper - lower`` (``>= 0``)."""
        return self.upper - self.lower

    @property
    def relative_width(self) -> float:
        """Enclosure width relative to the upper bound ``width / max(|upper|, tiny)``."""
        return self.width / max(abs(self.upper), _TINY)

    @property
    def is_sound(self) -> bool:
        """Whether the sandwich is well-formed (``0 <= lower <= upper <= total``)."""
        return -_TINY <= self.lower <= self.upper + _TINY <= self.total + 1.0 + _TINY

    def contains(self, count: float) -> bool:
        """Whether a candidate ``count`` lies inside the certified enclosure."""
        return self.lower - _TINY <= count <= self.upper + _TINY

    def seal(self, problem: ModelCountProblem | None = None) -> Cert:
        r"""Seal this enclosure into a tamper-evident, Lean-checkable v1 certificate.

        Thin delegate to :func:`omnibias.logic.model_count.proof.seal_count_certificate`
        (imported lazily to avoid an import cycle). Pass ``problem`` to attach the exact
        inclusion-exclusion count identity when the enclosure is tight, unweighted, and small;
        otherwise the sealed certificate carries the enclosed-quantity sign obligation.
        """
        from omnibias.logic.model_count.proof import seal_count_certificate

        return seal_count_certificate(self, problem=problem)


__all__ = ["CountCertificate"]
