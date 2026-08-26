# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-26: net-to-annihilator export."""

from __future__ import annotations

import pytest
from omnibias.holonomic._core.export import (
    DISCLAIMER,
    AnnihilatorExport,
    export_annihilator,
    export_skill,
    honesty_payload,
    ore_from_dict,
    ore_to_dict,
    source_imports_no_backend,
    worked_example,
)
from omnibias.holonomic._core.layer import d_minus_1


def test_g1_round_trip() -> None:
    ex = worked_example()
    assert ex["match"] is True
    assert ex["order"] == 1
    op = d_minus_1()
    assert ore_to_dict(ore_from_dict(ore_to_dict(op))) == ore_to_dict(op)


def test_g2_flags_false() -> None:
    report = export_skill()
    assert report["flags_false"] is True
    exported = export_annihilator(d_minus_1())
    assert exported.theorem_prover_verified is False
    assert exported.mathlib_verified is False
    with pytest.raises(ValueError, match="future-earned"):
        AnnihilatorExport(ore={}, lean_path=None, theorem_prover_verified=True)


def test_g3_forbidden_lean() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        export_annihilator(d_minus_1(), emit_lean=True, lean_claim="a continuum theorem")
    assert honesty_payload()["continuum_claim"] is False
    assert "finite rational" in DISCLAIMER


def test_g4_no_torch_jax() -> None:
    assert source_imports_no_backend() is True
