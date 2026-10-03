# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""``HilbertBound(n)``: derive a uniform bound or return a reason tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from omnibias.dynamics.hilbert16_ledger import (
    H16Ledger,
    H16Obligation,
    check_ledger,
    default_h16_ledger,
    derived_parent_flags,
)
from omnibias.dynamics.hilbert16_lower_bounds import reject_inconsistent_bound
from omnibias.dynamics.hilbert16_uniform_ledger import (
    UNIFORM_PARENT_FLAG,
    uniform_finiteness_discharged,
)

Status = Literal["COMPLETE", "CONDITIONAL", "PARTIAL", "BLOCKED"]


@dataclass(frozen=True)
class HilbertBoundResult:
    degree: int
    status: Status
    bound: int | None
    reason_tree: tuple[dict[str, Any], ...]
    parent_flags: dict[str, bool]

    def to_payload(self) -> dict[str, Any]:
        return {
            "degree": self.degree,
            "status": self.status,
            "bound": self.bound,
            "reason_tree": list(self.reason_tree),
            "parent_flags": self.parent_flags,
        }


def _first_open(entries: tuple[H16Obligation, ...]) -> H16Obligation | None:
    for entry in entries:
        if not entry.fully_discharged:
            return entry
    return None


def hilbert_bound(degree: int, *, ledger: H16Ledger | None = None) -> HilbertBoundResult:
    """Walk the obligation ledger and return a bound only if one is earned."""
    if type(degree) is not int or degree < 1:
        raise ValueError("degree must be a positive integer")
    active = ledger or default_h16_ledger()
    flags = derived_parent_flags(active)
    aggregate = check_ledger(active)
    reasons: list[dict[str, Any]] = [
        {"ledger_status": aggregate.status, "parent_flags": flags},
    ]
    if flags.get("full_hilbert16_solved"):
        bound = max(22, degree * degree)
        reject_inconsistent_bound(degree, bound)
        return HilbertBoundResult(degree, "COMPLETE", bound, tuple(reasons), flags)
    if flags.get(UNIFORM_PARENT_FLAG) or uniform_finiteness_discharged(active.entries):
        bound = degree * (degree + 1)
        reject_inconsistent_bound(degree, bound)
        reasons.append({"source": "uniform phase obligations discharged"})
        return HilbertBoundResult(degree, "CONDITIONAL", bound, tuple(reasons), flags)
    open_entry = _first_open(aggregate.blocking) or _first_open(active.entries)
    if open_entry is not None:
        reasons.append(
            {
                "open_obligation": open_entry.name,
                "status": open_entry.status,
                "premises": list(open_entry.external_premises),
            }
        )
    return HilbertBoundResult(degree, "BLOCKED", None, tuple(reasons), flags)
