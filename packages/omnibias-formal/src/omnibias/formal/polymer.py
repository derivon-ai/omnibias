# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Named polymer-coordination certificates for locked integer majorants.

A ``polymer`` payload names one locked family (``backtrack_4``,
``crude_4``, ``first_step_4``).
The Mathlib bridge re-derives the integers and emits Lean that applies the
matching ``OmnibiasAnalytic.Check`` Polymer theorem.

Each family is a finite arithmetic identity on a named dimension. It is
not a continuum gauge claim and not Osterwalder-Seiler.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from omnibias.core.proof.certificate import Cert, make_certificate

LEGAL_POLYMER_FAMILIES: tuple[str, ...] = ("backtrack_4", "crude_4", "first_step_4")

PolymerFamily = Literal["backtrack_4", "crude_4", "first_step_4"]


def polymer_backtrack(spacetime_dim: int) -> int:
    """``C = 3(2d - 3)``. Matches ``polymer_coordination_backtrack``."""
    if spacetime_dim < 2:
        raise ValueError(f"spacetime_dim must be >= 2, got {spacetime_dim}")
    return 3 * (2 * int(spacetime_dim) - 3)


def polymer_crude(spacetime_dim: int) -> int:
    """``C = 8(d - 1)``. Matches ``polymer_coordination``."""
    if spacetime_dim < 2:
        raise ValueError(f"spacetime_dim must be >= 2, got {spacetime_dim}")
    return 8 * (int(spacetime_dim) - 1)


def polymer_first_step(spacetime_dim: int) -> int:
    """``A = 4(2d - 3)``. Matches ``polymer_first_step`` in geometry."""
    if spacetime_dim < 2:
        raise ValueError(f"spacetime_dim must be >= 2, got {spacetime_dim}")
    return 4 * (2 * int(spacetime_dim) - 3)


_LEAN_THMS: dict[str, str] = {
    "backtrack_4": "polymer_backtrack_coord_4",
    "crude_4": "polymer_crude_coord_4",
    "first_step_4": "polymer_first_step_4",
}

_CLAIMS: dict[str, str] = {
    "backtrack_4": "backtrack polymer coordination at d=4 is 15",
    "crude_4": "crude polymer coordination at d=4 is 24",
    "first_step_4": "first-step polymer coordination at d=4 is 20",
}

_LOCKED_VALUE: dict[str, int] = {
    "backtrack_4": 15,
    "crude_4": 24,
    "first_step_4": 20,
}


def locked_polymer_matches(payload: Mapping[str, Any]) -> bool:
    """``True`` iff ``payload`` carries the locked integer for its family."""
    family = payload.get("family")
    if family not in _LOCKED_VALUE:
        return False
    value = payload.get("value")
    return isinstance(value, int) and not isinstance(value, bool) and value == _LOCKED_VALUE[family]


def family_facts_hold(family: str) -> bool:
    """Re-derive the locked coordination identity over ``int``."""
    if family == "backtrack_4":
        return polymer_backtrack(4) == 15 and polymer_backtrack(4) < polymer_crude(4)
    if family == "crude_4":
        return polymer_crude(4) == 24
    if family == "first_step_4":
        return (
            polymer_first_step(4) == 20
            and polymer_backtrack(4) < polymer_first_step(4)
        )
    return False


def lean_polymer_theorem(family: str) -> str:
    """Lean Check theorem applied by the ``polymer`` generator."""
    try:
        return _LEAN_THMS[family]
    except KeyError as exc:
        raise ValueError(f"unknown polymer family {family!r}") from exc


def polymer_certificate(
    family: PolymerFamily,
    *,
    honesty: Mapping[str, bool] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Cert:
    """Seal a planted ``polymer`` certificate for the Mathlib bridge."""
    if family not in LEGAL_POLYMER_FAMILIES:
        raise ValueError(
            f"unknown polymer family {family!r}; expected one of {LEGAL_POLYMER_FAMILIES}"
        )
    return make_certificate(
        claim=_CLAIMS[family],
        payload={
            "type": "polymer",
            "family": family,
            "value": _LOCKED_VALUE[family],
        },
        honesty=honesty,
        meta=meta,
    )


__all__ = [
    "LEGAL_POLYMER_FAMILIES",
    "family_facts_hold",
    "lean_polymer_theorem",
    "locked_polymer_matches",
    "polymer_backtrack",
    "polymer_certificate",
    "polymer_crude",
    "polymer_first_step",
]
