# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-12: holonomic layer."""

from __future__ import annotations

import ast
from fractions import Fraction
from pathlib import Path

import pytest
from omnibias.holonomic._core.layer import (
    DISCLAIMER,
    HolonomicLayerSpec,
    d2_plus_1,
    d_minus_1,
    holonomic_jet,
    honesty_payload,
    sin_skill,
    worked_example,
)


def test_g1_exp_jet() -> None:
    ex = worked_example()
    assert ex["exp_err"] < 1e-12
    assert ex["exp_jet"] == (Fraction(1),) * 6
    jet = holonomic_jet(d_minus_1(), (1,), 0, 5)
    assert jet == (Fraction(1),) * 6


def test_g1_sin_jet() -> None:
    jet = holonomic_jet(d2_plus_1(), (0, 1), 0, 3)
    assert jet[2] == 0
    assert jet[3] == -1


def test_g2_sin_skill() -> None:
    report = sin_skill()
    assert report["g2_earned"] is True
    assert report["multiple_of_d2_plus_1"] is True
    assert report["residual_ok"] is True


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["nonlinear_pde_claimed_dfinite"] is False
    assert payload["theorem_prover_verified"] is False
    assert "not a general PINN" in DISCLAIMER


def test_g4_layer_has_no_backend_imports() -> None:
    path = Path(__file__).resolve().parents[1] / "src/omnibias/holonomic/_core/layer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    banned = {"torch", "jax", "tensorflow", "keras"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in banned
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned


def test_domain() -> None:
    with pytest.raises(ValueError, match="initial jet"):
        holonomic_jet(d2_plus_1(), (0,), 0, 2)
    spec = HolonomicLayerSpec(max_order=0)
    from omnibias.holonomic._core.layer import fit_holonomic_layer

    with pytest.raises(ValueError, match="max_order"):
        fit_holonomic_layer([1, 1, 1], spec=spec)
