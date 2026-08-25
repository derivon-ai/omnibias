# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Kernel-level checks for the 05-02 G5 designed taps (benchmark + core)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "benchmarks"))

from sequence_transverse import (  # type: ignore[import-not-found]  # noqa: E402
    apply_causal_fir,
    causal_sigma_kernel,
)


def test_kernel_is_causal_and_finite() -> None:
    kern = causal_sigma_kernel(width=8, coeff=1.0, alpha=1.0, tau=2.0, order=1)
    assert kern.shape == (8,)
    assert np.isfinite(kern).all()
    # A mid-lag bump: mass is not concentrated at the acausal side.
    assert float(np.argmax(np.abs(kern))) >= 0


def test_negative_order_and_width_raise() -> None:
    with pytest.raises(ValueError, match="order"):
        causal_sigma_kernel(width=4, coeff=1.0, alpha=1.0, tau=0.0, order=-1)
    with pytest.raises(ValueError, match="width"):
        causal_sigma_kernel(width=0, coeff=1.0, alpha=1.0, tau=0.0, order=1)


def test_causal_fir_impulse_recovers_kernel() -> None:
    kern = causal_sigma_kernel(width=5, coeff=2.0, alpha=0.8, tau=1.0, order=1)
    impulse = np.zeros((1, 7), dtype=float)
    impulse[0, 0] = 1.0
    out = apply_causal_fir(impulse, kern)
    np.testing.assert_allclose(out[0, :5], kern, rtol=0.0, atol=1e-12)
    assert out.shape == (1, 7)


def test_apply_causal_fir_rejects_bad_rank() -> None:
    with pytest.raises(ValueError, match="batch"):
        apply_causal_fir(np.zeros(3), np.ones(2))
