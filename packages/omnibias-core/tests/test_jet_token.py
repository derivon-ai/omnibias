# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-02 / 09-19: jet-token mix and jet distillation."""

from __future__ import annotations

import pytest
from omnibias.core.jet_token import (
    DISCLAIMER,
    JetTokenConfig,
    check_full_width,
    distill_skill,
    honesty_payload,
    jet_token_skill,
    recover_tanh_scale,
    ssl_flip_residual,
    worked_example,
)


def test_g1_worked_mix() -> None:
    ex = worked_example()
    assert ex["value_err"] < 1e-12
    assert ex["deriv_err"] < 1e-12
    rec = recover_tanh_scale(0.5)
    assert rec["a"] == 1.0
    assert rec["loss"] < 1e-12
    assert ssl_flip_residual(0.7) == 0.0


def test_g2_jet_token_beats_value() -> None:
    report = jet_token_skill()
    assert report["below_1e3"] is True
    assert report["beats_value"] is True
    assert float(report["skill"]) > 0.0


def test_g2_distill_beats_value() -> None:
    report = distill_skill()
    assert report["below_1e3"] is True
    assert report["beats_value"] is True


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["imagenet_claim"] is False
    assert "not ImageNet" in DISCLAIMER


def test_reject_full_parameter_map() -> None:
    with pytest.raises(ValueError, match="allow_full"):
        check_full_width(JetTokenConfig(n_directions=2, width=8), n_params=16)
    check_full_width(JetTokenConfig(n_directions=2, width=8, allow_full=True), n_params=16)
