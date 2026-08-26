# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pairing collapse: weak zero on a pack is not a strong solution."""

from __future__ import annotations

import pytest
from omnibias.core.collapse import (
    PAIRING_SPEC,
    are_distinct,
    get_collapse,
    pairing_collapse,
    reset_collapse_registry,
)
from omnibias.core.collapse.pairing import pairing_value


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    reset_collapse_registry()
    yield
    reset_collapse_registry()


def test_pairing_is_registered_and_distinct() -> None:
    assert get_collapse("pairing") == PAIRING_SPEC
    for name in ("bias", "temperature", "enclosure", "verdict", "identity", "winding"):
        assert are_distinct(PAIRING_SPEC, get_collapse(name)).distinct


def test_zero_residual_collapses_against_any_pack() -> None:
    verdict = pairing_collapse((0,), ((1,), (0, 1), (1, 0, 1)))
    assert verdict.proved
    assert verdict.outcome.surviving == "weak_residual_on_pack"
    assert verdict.outcome.honesty["not_a_strong_solution"] is True
    assert verdict.outcome.honesty["continuum_parent_inferred"] is False


def test_constant_residual_against_mass_is_disproved() -> None:
    assert pairing_value((1,), (1,)) == 2
    verdict = pairing_collapse((1,), ((1,),))
    assert verdict.disproved


def test_odd_residual_against_even_tests_is_weakly_zero() -> None:
    verdict = pairing_collapse((0, 1), ((1,), (1, 0, 1)))
    assert verdict.proved
    assert verdict.outcome.honesty["not_a_strong_solution"] is True


def test_empty_pack_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one test"):
        pairing_collapse((1,), ())
