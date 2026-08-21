# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-09: sheaf-atlas cocycle residual."""

from __future__ import annotations

from omnibias.geometry.atlas.cocycle import (
    DISCLAIMER,
    honesty_payload,
    two_interval_poisson,
    worked_example,
)


def test_g1_algebra() -> None:
    ex = worked_example()
    assert ex["residual"] < 1e-12
    assert abs(ex["composed_deriv"] - 1.0) < 1e-12
    assert ex["buggy_deriv_gap"] > 1e-3


def test_g2_poisson_overlap() -> None:
    report = two_interval_poisson()
    assert report["nonempty_overlap"] is True
    assert float(report["skill_vs_zero"]) > 0.0
    assert report["below_1e6"] is True
    assert report["ignored_cocycle"] is False


def test_g3_honesty() -> None:
    assert honesty_payload()["temperature_collapse_used"] is False
    assert honesty_payload(beta=1e9)["temperature_collapse_used"] is True
    assert honesty_payload()["sheaf_cohomology_theorem"] is False
    assert "not a sheaf-cohomology theorem" in DISCLAIMER
