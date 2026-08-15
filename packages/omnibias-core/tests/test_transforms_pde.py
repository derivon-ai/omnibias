# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Named PDE transforms G1 (theory 02-13). Vocabulary, not 03-11 search."""

from __future__ import annotations

from omnibias.core.transforms_pde import (
    TransformKind,
    cole_hopf_from_heat_phi,
    darboux_dress,
    miura_v,
    named_cole_hopf,
    verify_transform,
)


def test_cole_hopf_worked_example() -> None:
    t = named_cole_hopf()
    assert t.kind is TransformKind.COLE_HOPF
    assert verify_transform(t, order=8)
    assert cole_hopf_from_heat_phi(0.0, 0.0) == -2.0


def test_miura_and_darboux_finite() -> None:
    assert miura_v(0.5, -0.2) == -0.2 + 0.25
    assert darboux_dress(2.0, 1.0, 0.3) == 0.3 - 1.0
