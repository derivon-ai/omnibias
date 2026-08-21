# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-11: Collapse-Net."""

from __future__ import annotations

import math

import pytest
from omnibias.core.collapse_net import (
    DISCLAIMER,
    CollapseNetConfig,
    collapse_net_forward,
    collapse_remainder,
    honesty_payload,
    sin_skill,
    worked_example,
)


def test_g1_remainder() -> None:
    ex = worked_example()
    assert ex["rel_err"] < 1e-6
    assert abs(ex["collapsed"] - 0.25) < 1e-12
    rem = collapse_remainder(0.0, config=CollapseNetConfig(order=1, delta=0.1))
    assert abs(rem - ex["remainder"]) < 1e-15


def test_g2_sin_skill() -> None:
    report = sin_skill()
    assert report["below_1e3"] is True
    assert report["skill_positive"] is True
    assert report["collapsed_no_worse"] is True


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["continuum_pde_claimed"] is False
    assert "not a continuum PDE" in DISCLAIMER


def test_modes_and_domain() -> None:
    cfg_st = CollapseNetConfig(mode="stencil", family="sin", delta=0.1)
    cfg_co = CollapseNetConfig(mode="collapsed", family="sin", delta=0.1)
    x = 0.3
    assert collapse_net_forward(x, config=cfg_co) == pytest.approx(math.cos(x))
    st = collapse_net_forward(x, config=cfg_st)
    expect = (math.sin(x + 0.1) - math.sin(x - 0.1)) / 0.2
    assert st == pytest.approx(expect)
    with pytest.raises(ValueError, match="delta"):
        collapse_net_forward(0.0, config=CollapseNetConfig(delta=0.0))
    with pytest.raises(ValueError, match="family"):
        collapse_net_forward(0.0, config=CollapseNetConfig(family="tanh"))
    with pytest.raises(ValueError, match="mode"):
        collapse_net_forward(0.0, config=CollapseNetConfig(mode="eval"))
