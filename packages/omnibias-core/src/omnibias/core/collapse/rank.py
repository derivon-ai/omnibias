# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Rank / syzygy collapse: an exact ``Q`` kernel, not a float singular value.

A numerical ``σ_min -> 0`` is only a proposer. The accept is an exact
integer nullspace whose vectors satisfy ``M v = 0`` over ``Q``. The
surviving object is a **syzygy** (an exact algebraic relation). A
float residual or a rank-revealing SVD is not a certificate.

An empty kernel **disproves** the existential claim "a nontrivial
relation exists" for this finite matrix. It does not certify a
holonomic identity of a special function unless the caller built that
determining matrix. Do not conflate with founding bias collapse
(``delta -> 0``) or temperature collapse (``beta -> inf``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.collapse.schema import (
    CollapseOutcome,
    CollapseSpec,
    add_registry_hook,
    default_honesty,
    get_collapse,
    register_collapse,
)
from omnibias.core.collapse.verdict import ObligationVerdict
from omnibias.core.proof.lift import integer_null_space, residual_identically_zero

RANK_SPEC = CollapseSpec(
    name="rank",
    parameter="sigma_min",
    limit="0",
    surviving_object="syzygy",
    failure="numerical rank drop without Q reconstruction",
    home="omnibias.core.collapse.rank",
    register="holonomic",
)


def _honesty() -> dict[str, bool]:
    payload = default_honesty(spec_name="rank")
    payload["rank_collapse"] = True
    payload["float_svd_is_proof"] = False
    payload["holonomic_special_function_claim"] = False
    return payload


def _require_integer_matrix(
    matrix: Sequence[Sequence[object]],
) -> list[list[int]]:
    rows: list[list[int]] = []
    width = 0
    for row in matrix:
        ints: list[int] = []
        for entry in row:
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise TypeError(
                    "rank collapse requires an exact integer matrix; "
                    "a float singular value is not a certificate"
                )
            ints.append(int(entry))
        if not ints:
            raise ValueError("matrix rows must be non-empty")
        if width == 0:
            width = len(ints)
        elif len(ints) != width:
            raise ValueError("matrix rows must share a width")
        rows.append(ints)
    if not rows:
        raise ValueError("matrix must be non-empty")
    return rows


@dataclass(frozen=True)
class RankReport:
    """Exact kernel plus the three-way verdict."""

    verdict: ObligationVerdict
    kernel: tuple[tuple[int, ...], ...]

    @property
    def proved(self) -> bool:
        return self.verdict.proved

    @property
    def disproved(self) -> bool:
        return self.verdict.disproved

    @property
    def blocked(self) -> bool:
        return self.verdict.blocked


def rank_collapse(matrix: Sequence[Sequence[object]]) -> RankReport:
    """Prove or disprove a nontrivial exact syzygy of ``matrix``."""

    rows = _require_integer_matrix(matrix)
    kernel = tuple(tuple(vec) for vec in integer_null_space(rows))
    zeros = [0] * len(rows)
    for vec in kernel:
        if not residual_identically_zero(rows, list(vec), zeros):
            outcome = CollapseOutcome(
                status="inconclusive",
                spec_name="rank",
                surviving=None,
                residual=None,
                detail="reconstructed kernel failed the exact Q check",
                honesty=_honesty(),
            )
            verdict = ObligationVerdict(
                status="BLOCKED",
                outcome=outcome,
                existential=True,
                evaluated=len(kernel),
                complete=False,
                detail=outcome.detail,
            )
            return RankReport(verdict=verdict, kernel=kernel)
    if not kernel:
        outcome = CollapseOutcome(
            status="excluded",
            spec_name="rank",
            surviving="DISPROVED",
            residual=None,
            detail="exact kernel is trivial; no syzygy in this matrix",
            honesty=_honesty(),
        )
        verdict = ObligationVerdict(
            status="DISPROVED",
            outcome=outcome,
            existential=True,
            evaluated=1,
            complete=True,
            detail=outcome.detail,
        )
        return RankReport(verdict=verdict, kernel=())
    outcome = CollapseOutcome(
        status="collapsed",
        spec_name="rank",
        surviving="syzygy",
        residual=None,
        detail=f"exact syzygy {kernel}",
        honesty=_honesty(),
    )
    verdict = ObligationVerdict(
        status="PROVED",
        outcome=outcome,
        existential=True,
        evaluated=len(kernel),
        complete=True,
        detail=outcome.detail,
    )
    return RankReport(verdict=verdict, kernel=kernel)


def _reseed() -> None:
    try:
        get_collapse("rank")
    except KeyError:
        register_collapse(RANK_SPEC)


add_registry_hook(_reseed)


__all__ = [
    "RANK_SPEC",
    "RankReport",
    "rank_collapse",
]
