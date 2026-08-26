# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Rank collapse: exact Q kernel only; a float SVD is not a proof."""

from __future__ import annotations

import pytest
from omnibias.core.collapse import (
    RANK_SPEC,
    are_distinct,
    get_collapse,
    rank_collapse,
    reset_collapse_registry,
)
from omnibias.core.proof.lift import residual_identically_zero


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    reset_collapse_registry()
    yield
    reset_collapse_registry()


def test_rank_is_registered_and_distinct() -> None:
    assert get_collapse("rank") == RANK_SPEC
    for name in (
        "bias",
        "temperature",
        "enclosure",
        "verdict",
        "identity",
        "winding",
        "pairing",
    ):
        assert are_distinct(RANK_SPEC, get_collapse(name)).distinct


def test_dependent_rows_yield_an_exact_syzygy() -> None:
    report = rank_collapse(((1, 2), (2, 4)))
    assert report.proved
    assert report.kernel
    zeros = [0, 0]
    for vec in report.kernel:
        assert residual_identically_zero(((1, 2), (2, 4)), list(vec), zeros)
    assert report.verdict.outcome.surviving == "syzygy"
    assert report.verdict.outcome.honesty["float_svd_is_proof"] is False
    assert report.verdict.outcome.honesty["holonomic_special_function_claim"] is False


def test_full_rank_has_no_syzygy() -> None:
    report = rank_collapse(((1, 0), (0, 1)))
    assert report.disproved
    assert report.kernel == ()


def test_float_matrix_is_refused() -> None:
    with pytest.raises(TypeError, match="float singular value"):
        rank_collapse(((1.0, 2.0), (2.0, 4.0)))
