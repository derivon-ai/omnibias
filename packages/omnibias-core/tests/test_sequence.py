# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Numpy-free checks for designed causal taps (theory 05-02)."""

from __future__ import annotations

import math

import pytest
from omnibias.core.sequence import causal_transverse_taps, leaky_integrator_init


def test_leaky_integrator_init_matches_ar1_tail() -> None:
    coeff, alpha, tau = leaky_integrator_init(0.95)
    assert coeff == pytest.approx(0.05)
    assert alpha == pytest.approx(-math.log(0.95))
    assert tau == 0.0


def test_leaky_integrator_init_rejects_closed_interval() -> None:
    with pytest.raises(ValueError, match="rho"):
        leaky_integrator_init(0.0)
    with pytest.raises(ValueError, match="rho"):
        leaky_integrator_init(1.0)


def test_order0_is_a_decreasing_logistic_tail() -> None:
    taps = causal_transverse_taps(width=8, coeff=1.0, alpha=0.5, tau=0.0, order=0)
    assert len(taps) == 8
    assert all(math.isfinite(v) for v in taps)
    assert all(taps[i] >= taps[i + 1] for i in range(len(taps) - 1))
    assert taps[0] == pytest.approx(0.5)


def test_order1_is_a_mid_lag_bump() -> None:
    taps = causal_transverse_taps(width=12, coeff=1.0, alpha=1.0, tau=4.0, order=1)
    peak = max(range(len(taps)), key=lambda i: abs(taps[i]))
    assert peak >= 2


def test_negative_order_and_width_raise() -> None:
    with pytest.raises(ValueError, match="order"):
        causal_transverse_taps(width=4, coeff=1.0, alpha=1.0, tau=0.0, order=-1)
    with pytest.raises(ValueError, match="width"):
        causal_transverse_taps(width=0, coeff=1.0, alpha=1.0, tau=0.0, order=0)
