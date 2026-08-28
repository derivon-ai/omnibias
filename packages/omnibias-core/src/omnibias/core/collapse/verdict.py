# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Obligation / verdict collapse: a sound residual enclosure decides.

Training (or any search) may propose a witness. The accept is *not*
a float loss going to zero. The accept is a sound enclosure of the
residual of a *finite* obligation collapsing onto the singleton
``{0}``.

* enclosure is ``{0}`` → ``PROVED``
* ``0`` lies outside the enclosure → ``DISPROVED``
* ``0`` is inside a positive-width enclosure → ``BLOCKED``
  (``Inconclusive`` / ``search_incomplete``). This is not falsity.

A float residual is refused. ``theorem_prover_verified`` stays false.
Continuum parents are never inferred. Do not conflate this with
founding bias collapse (``delta -> 0``), temperature collapse
(``beta -> inf``), or Enclosure Collapse (``width -> 0`` of a sound
enclosure, yielding a point plus a proof).
The surviving object here is a :class:`ObligationVerdict`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from omnibias.core.collapse.schema import (
    CollapseOutcome,
    CollapseSpec,
    add_registry_hook,
    default_honesty,
    get_collapse,
    register_collapse,
    require_sound_enclosure,
)
from omnibias.core.verified.interval import Interval

VerdictStatus = Literal["PROVED", "DISPROVED", "BLOCKED"]

VERDICT_SPEC = CollapseSpec(
    name="verdict",
    parameter="witness",
    limit="{0}",
    surviving_object="verdict",
    failure="Inconclusive",
    home="omnibias.core.collapse.verdict",
    register="formal",
)


def _honesty() -> dict[str, bool]:
    payload = default_honesty(spec_name="verdict")
    payload["verdict_collapse"] = True
    return payload


def is_singleton_zero(residual: Interval) -> bool:
    """``True`` iff the enclosure is the exact point ``{0}``."""

    return residual.lo == 0.0 and residual.hi == 0.0


@dataclass(frozen=True)
class ObligationVerdict:
    """Adjudication of one residual or a finite residual family."""

    status: VerdictStatus
    outcome: CollapseOutcome
    existential: bool
    evaluated: int
    complete: bool
    detail: str

    @property
    def proved(self) -> bool:
        return self.status == "PROVED"

    @property
    def disproved(self) -> bool:
        return self.status == "DISPROVED"

    @property
    def blocked(self) -> bool:
        return self.status == "BLOCKED"

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "existential": self.existential,
            "evaluated": self.evaluated,
            "complete": self.complete,
            "detail": self.detail,
            "outcome": self.outcome.to_payload(),
        }


def _outcome(status: Literal["collapsed", "excluded", "inconclusive"], residual: Interval) -> CollapseOutcome:
    if status == "collapsed":
        surviving: str | None = "PROVED"
        detail = "sound enclosure is the singleton {0}"
    elif status == "excluded":
        surviving = "DISPROVED"
        detail = "0 lies outside the sound residual enclosure"
    else:
        surviving = None
        detail = (
            "0 is inside the enclosure and the enclosure is not {0}; "
            "Inconclusive, not false"
        )
    return CollapseOutcome(
        status=status,
        spec_name="verdict",
        surviving=surviving,
        residual=residual,
        detail=detail,
        honesty=_honesty(),
    )


def adjudicate_residual(
    residual: object,
    *,
    existential: bool = True,
) -> ObligationVerdict:
    """Decide one sound residual enclosure of a finite obligation."""

    boxed = require_sound_enclosure(residual)
    if is_singleton_zero(boxed):
        outcome = _outcome("collapsed", boxed)
        return ObligationVerdict(
            status="PROVED",
            outcome=outcome,
            existential=existential,
            evaluated=1,
            complete=True,
            detail=outcome.detail,
        )
    if not boxed.contains_zero():
        outcome = _outcome("excluded", boxed)
        return ObligationVerdict(
            status="DISPROVED",
            outcome=outcome,
            existential=existential,
            evaluated=1,
            complete=True,
            detail=outcome.detail,
        )
    outcome = _outcome("inconclusive", boxed)
    return ObligationVerdict(
        status="BLOCKED",
        outcome=outcome,
        existential=existential,
        evaluated=1,
        complete=False,
        detail=outcome.detail,
    )


def search_residuals(
    residuals: Sequence[object],
    *,
    existential: bool = True,
    complete: bool = False,
) -> ObligationVerdict:
    """Search a finite family of residual enclosures.

    Existential hit: some residual collapses to ``{0}`` → ``PROVED``.
    Universal hit: some residual excludes ``0`` → ``DISPROVED``.
    A complete existential miss is ``BLOCKED`` (no witness in *this*
    family), never a parent-false claim. Any inconclusive enclosure
    makes the walk ``search_incomplete``.
    """

    boxes = [require_sound_enclosure(item) for item in residuals]
    if not boxes:
        outcome = CollapseOutcome(
            status="inconclusive",
            spec_name="verdict",
            surviving=None,
            residual=None,
            detail="empty family; search_incomplete",
            honesty=_honesty(),
        )
        return ObligationVerdict(
            status="BLOCKED",
            outcome=outcome,
            existential=existential,
            evaluated=0,
            complete=False,
            detail=outcome.detail,
        )

    inconclusive = False
    last = adjudicate_residual(boxes[0], existential=existential)
    for index, boxed in enumerate(boxes, start=1):
        last = adjudicate_residual(boxed, existential=existential)
        if existential and last.proved:
            return replace(last, evaluated=index)
        if not existential and last.disproved:
            return replace(last, evaluated=index)
        if last.blocked:
            inconclusive = True

    if inconclusive or not complete:
        outcome = CollapseOutcome(
            status="inconclusive",
            spec_name="verdict",
            surviving=None,
            residual=last.outcome.residual,
            detail="search_incomplete; Inconclusive is not falsity",
            honesty=_honesty(),
        )
        return ObligationVerdict(
            status="BLOCKED",
            outcome=outcome,
            existential=existential,
            evaluated=len(boxes),
            complete=False,
            detail=outcome.detail,
        )

    if existential:
        outcome = CollapseOutcome(
            status="excluded",
            spec_name="verdict",
            surviving="BLOCKED",
            residual=last.outcome.residual,
            detail="no witness in this enumerated family; not a parent claim",
            honesty=_honesty(),
        )
        return ObligationVerdict(
            status="BLOCKED",
            outcome=outcome,
            existential=True,
            evaluated=len(boxes),
            complete=True,
            detail=outcome.detail,
        )

    outcome = CollapseOutcome(
        status="collapsed",
        spec_name="verdict",
        surviving="PROVED",
        residual=last.outcome.residual,
        detail="finite universal: every enumerated residual is {0}",
        honesty=_honesty(),
    )
    return ObligationVerdict(
        status="PROVED",
        outcome=outcome,
        existential=False,
        evaluated=len(boxes),
        complete=True,
        detail=outcome.detail,
    )


def _reseed() -> None:
    try:
        get_collapse("verdict")
    except KeyError:
        register_collapse(VERDICT_SPEC)


add_registry_hook(_reseed)


__all__ = [
    "ObligationVerdict",
    "VERDICT_SPEC",
    "VerdictStatus",
    "adjudicate_residual",
    "is_singleton_zero",
    "search_residuals",
]
