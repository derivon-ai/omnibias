# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CausalTransverseFilter unit checks (theory 05-02)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.core.sequence import causal_transverse_taps
from omnibias.torch.sequence import CausalTransverseFilter


def test_from_leaky_integrator_taps_match_core() -> None:
    filt = CausalTransverseFilter.from_leaky_integrator(0.95, width=8, dtype=torch.float64)
    taps = filt.taps().detach().cpu().numpy()
    expected = np.asarray(
        causal_transverse_taps(
            width=8,
            coeff=float(filt.coeff.detach()),
            alpha=float(filt.timescale().detach()),
            tau=float(filt.tau.detach()),
            order=0,
        )
    )
    np.testing.assert_allclose(taps, expected, rtol=0.0, atol=1e-12)


def test_impulse_recovers_taps() -> None:
    filt = CausalTransverseFilter(
        order=0, alpha=0.2, width=5, coeff=0.4, tau=0.0, dtype=torch.float64
    )
    impulse = torch.zeros(1, 7, dtype=torch.float64)
    impulse[0, 0] = 1.0
    out = filt(impulse).detach()
    taps = filt.taps().detach()
    torch.testing.assert_close(out[0, :5], taps, rtol=0.0, atol=1e-12)
    assert out.shape == (1, 7)


def test_rejects_bad_order_width_alpha() -> None:
    with pytest.raises(ValueError, match="order"):
        CausalTransverseFilter(order=-1, alpha=0.1, width=4)
    with pytest.raises(ValueError, match="width"):
        CausalTransverseFilter(order=0, alpha=0.1, width=0)
    with pytest.raises(ValueError, match="alpha"):
        CausalTransverseFilter(order=0, alpha=0.0, width=4)
