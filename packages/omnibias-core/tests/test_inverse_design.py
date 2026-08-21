# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-22: Newton-on-input inverse design."""

from __future__ import annotations

import pytest
from omnibias.core.inverse_design import (
    DISCLAIMER,
    honesty_payload,
    invert_input,
    invert_skill,
    worked_example,
)


def test_g1_tanh_half() -> None:
    ex = worked_example()
    assert ex["residual"] < 1e-12
    assert ex["abs_x_err"] < 1e-12
    assert ex["f_prime"] == pytest.approx(1.5, abs=1e-12)


def test_g2_skill() -> None:
    report = invert_skill()
    assert report["g2_earned"] is True
    assert report["saturated_raised"] is True
    assert report["all_finite"] is True
    assert float(report["median_err"]) < 1e-10  # type: ignore[arg-type]


def test_g2_saturated_raises() -> None:
    with pytest.raises(ValueError, match="saturat"):
        invert_input(None, 0.999, 0.0)


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["global_inverse_claimed"] is False
    assert payload["layer_invert_claimed"] is False
    assert payload["theorem_prover_verified"] is False
    assert "not 08-03" in DISCLAIMER
