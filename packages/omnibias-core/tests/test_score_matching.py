# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-21: exact Hyvärinen score matching."""

from __future__ import annotations

from omnibias.core.score_matching import (
    DISCLAIMER,
    exact_div_variance,
    honesty_payload,
    score_matching_skill,
    worked_example,
)


def test_g1_div_neg_x() -> None:
    ex = worked_example()
    assert abs(ex["div"] + 1.0) < 1e-12
    assert ex["abs_div_err"] < 1e-12
    assert abs(ex["hyvarinen"] + 0.5) < 1e-12


def test_g2_skill() -> None:
    report = score_matching_skill()
    assert report["g2_earned"] is True
    assert report["reached"] >= 3
    assert max(report["exact_div_variance"]) == 0.0  # type: ignore[arg-type]
    assert report["hutchinson_exact_path_variance"] == 0.0


def test_exact_div_variance_preserves_a_constant_binary64_stream() -> None:
    """Exact divergence is constant even when its floating sum would round."""
    assert exact_div_variance([-1.0 / 3.0] * 64) == 0.0


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["cnf_div_claimed_new"] is False
    assert payload["theorem_prover_verified"] is False
    assert "prior art" in DISCLAIMER
