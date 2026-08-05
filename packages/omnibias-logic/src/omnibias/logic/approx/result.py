# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The result type for the **statistical, NOT worst-case sound** counters.

:class:`ApproxCount` is deliberately a *different* type from
:class:`~omnibias.logic.model_count.certificate.CountCertificate` so a statistical estimate
can never be mistaken for a rigorous enclosure. Its interval carries a **probabilistic /
coverage** guarantee (an ``(epsilon, delta)`` PAC bound, or a distribution-dependent
conformal coverage level) -- not a rigorous bracket that provably contains the true count.
The ``worst_case_sound`` flag is hard-wired ``False`` and refuses to be constructed ``True``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The one-line contract every statistical estimate carries.
NOT_SOUND_DISCLAIMER = (
    "STATISTICAL estimate: probabilistic / coverage guarantee only, NOT a worst-case-sound "
    "enclosure. Do not substitute for a CountCertificate or a CountResult."
)


@dataclass(frozen=True)
class ApproxCount:
    r"""A statistical model-count estimate with a probabilistic (NOT sound) interval.

    Attributes
    ----------
    estimate:
        The point estimate of the (weighted) model count.
    interval:
        A ``(lo, hi)`` interval whose guarantee is **statistical**: an ``(epsilon, delta)``
        PAC bracket for the hashing estimator, or a marginal-coverage interval for the
        conformal wrapper. It is *not* a rigorous enclosure -- the true count can fall outside
        it with the stated failure probability.
    method:
        The estimator that produced it (e.g. ``"xor_hashing"``, ``"split_conformal"``).
    epsilon, delta:
        The multiplicative tolerance / failure probability of an ``(epsilon, delta)``
        estimator, when applicable.
    confidence:
        The target coverage level ``1 - alpha`` for a conformal interval, when applicable.
    worst_case_sound:
        Hard-wired ``False``. Constructing it ``True`` raises -- soundness cannot be forged
        onto a statistical estimate.
    """

    estimate: float
    interval: tuple[float, float]
    method: str
    epsilon: float | None = None
    delta: float | None = None
    confidence: float | None = None
    worst_case_sound: bool = False

    def __post_init__(self) -> None:
        if self.worst_case_sound:
            raise ValueError(
                "ApproxCount is a statistical estimate and can never be worst_case_sound; "
                "use CountCertificate / count_enclosure for a rigorous bracket."
            )

    @property
    def lower(self) -> float:
        """The lower end of the (statistical) interval."""
        return self.interval[0]

    @property
    def upper(self) -> float:
        """The upper end of the (statistical) interval."""
        return self.interval[1]

    @property
    def disclaimer(self) -> str:
        """The non-sound contract string (see :data:`NOT_SOUND_DISCLAIMER`)."""
        return NOT_SOUND_DISCLAIMER

    def contains(self, count: float) -> bool:
        """Whether ``count`` lies in the interval -- a *statistical*, not rigorous, check."""
        return self.interval[0] <= count <= self.interval[1]


__all__ = ["ApproxCount", "NOT_SOUND_DISCLAIMER"]
