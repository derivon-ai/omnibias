# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Depth-causal local jet algebra (theory 08-03)."""

from __future__ import annotations

import math

import pytest
from omnibias.core.local_jet import (
    LocalJetConfig,
    LocalJetForbidden,
    invert_sigma,
    mlp_param_count,
    reject_local_jet_flood,
    require_invertible_sigma,
)


def test_invert_tanh_and_sigmoid() -> None:
    assert invert_sigma("tanh", 0.0) == pytest.approx(0.0)
    assert invert_sigma("tanh", math.tanh(0.5)) == pytest.approx(0.5)
    assert invert_sigma("sigmoid", 0.5) == pytest.approx(0.0)


def test_invert_gelu_raises() -> None:
    with pytest.raises(ValueError, match="strictly monotone"):
        require_invertible_sigma("gelu")
    with pytest.raises(ValueError, match="strictly monotone"):
        invert_sigma("gelu", 0.1)


def test_invert_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="artanh"):
        invert_sigma("tanh", 1.0)
    with pytest.raises(ValueError, match="logit"):
        invert_sigma("sigmoid", 0.0)


def test_g1_forbid_flood() -> None:
    with pytest.raises(LocalJetForbidden, match="allow_full"):
        reject_local_jet_flood(4, 4, allow_full=False)
    reject_local_jet_flood(4, 4, allow_full=True)
    reject_local_jet_flood(1, 2, allow_full=False)


def test_mlp_param_count() -> None:
    assert mlp_param_count(((1, None), (1, None))) == 2
    assert mlp_param_count(((8, 8), (8, None))) == 24


def test_config_rejects_bad_variant() -> None:
    with pytest.raises(ValueError, match="variant"):
        LocalJetConfig(variant="backprop")  # type: ignore[arg-type]
