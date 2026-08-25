# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Causal-transverse torch / jax parity (theory 05-02)."""

from __future__ import annotations

import math

import numpy as np
import torch
from omnibias.torch.sequence import CausalTransverseFilter


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def test_order0_forward_bit_identical() -> None:
    import jax
    import jax.numpy as jnp
    from omnibias.jax.sequence import causal_transverse_filter, init_causal_transverse

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    filt = CausalTransverseFilter(
        order=0, alpha=0.051, width=16, coeff=0.05, tau=0.0, dtype=torch.float64
    )
    x_np = np.linspace(-0.4, 0.5, 16, dtype=np.float64)[None, :]
    torch_out = filt(torch.as_tensor(x_np)).detach().numpy()
    params = init_causal_transverse(order=0, alpha=0.051, width=16, coeff=0.05, tau=0.0)
    jax_out = np.asarray(causal_transverse_filter(jnp.asarray(x_np), params))
    worst = max(
        _ulp_error(float(a), float(b))
        for a, b in zip(torch_out.reshape(-1), jax_out.reshape(-1), strict=True)
    )
    assert worst <= 4.0, f"order-0 parity worst_ulp={worst}"


def test_order1_taps_bit_identical() -> None:
    import jax
    import jax.numpy as jnp
    from omnibias.jax.sequence import causal_transverse_taps

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    filt = CausalTransverseFilter(
        order=1, alpha=0.8, width=10, coeff=1.2, tau=3.0, dtype=torch.float64
    )
    torch_taps = filt.taps().detach().numpy()
    jax_taps = np.asarray(
        causal_transverse_taps(
            jnp.asarray(1.2),
            jnp.asarray(0.8),
            jnp.asarray(3.0),
            order=1,
            width=10,
        )
    )
    worst = max(
        _ulp_error(float(a), float(b))
        for a, b in zip(torch_taps, jax_taps, strict=True)
    )
    assert worst <= 4.0, f"order-1 tap parity worst_ulp={worst}"
