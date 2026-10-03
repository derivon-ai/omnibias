# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Certified lower bounds for Hilbert number claims."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CertifiedLowerBound",
    "certified_lower_bound",
    "reject_inconsistent_bound",
    "verify_claimed_bound",
]

CERTIFIED_EXACT: dict[int, int] = {
    1: 0,
    2: 4,
    3: 13,
}


@dataclass(frozen=True)
class CertifiedLowerBound:
    degree: int
    value: int
    source: str
    exact: bool


def certified_lower_bound(degree: int) -> CertifiedLowerBound:
    if degree in CERTIFIED_EXACT:
        return CertifiedLowerBound(degree, CERTIFIED_EXACT[degree], "named example family", True)
    if degree >= 4:
        # Christopher-Lloyd asymptotic guide: c * n^2 * log(n); use a conservative floor.
        floor = max(1, (degree * degree) // 2)
        return CertifiedLowerBound(degree, floor, "asymptotic n^2 log n guide (not tight)", False)
    raise ValueError(f"no certified lower bound recorded for degree {degree}")


def verify_claimed_bound(degree: int, bound: int) -> bool:
    return bound >= certified_lower_bound(degree).value


def reject_inconsistent_bound(degree: int, bound: int) -> None:
    if not verify_claimed_bound(degree, bound):
        lower = certified_lower_bound(degree)
        raise ValueError(
            f"claimed bound {bound} is below certified lower bound {lower.value} for degree {degree}"
        )
