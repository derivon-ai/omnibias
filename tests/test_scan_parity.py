# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bias-scan torch/jax parity (theory 01-02 G3)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omnibias.core.scan import BankSpec


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def test_g3_torch_jax_bit_identical_response() -> None:
    import jax
    import jax.numpy as jnp
    from omnibias.jax.scan import bias_scan, init_bias_scan
    from omnibias.torch.scan import BiasScan

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    bank = BankSpec.uniform(-1.0, 1.0, 5)
    unit = BiasScan(
        2,
        bank,
        template="grad",
        base="tanh",
        learnable_offsets=False,
        learnable_scales=False,
        readout="response",
        dtype=torch.float64,
    )
    z_np = np.array([[0.0, 0.0], [0.3, -0.25], [0.5, 0.5]], dtype=np.float64)
    torch_out = unit(torch.as_tensor(z_np)).detach().numpy()
    act, offsets, scales, tmpl, _taps = init_bias_scan(2, bank, template="grad", base="tanh")
    jax_out = np.asarray(
        bias_scan(
            jnp.asarray(z_np),
            offsets,
            scales,
            tmpl,
            act,
            readout="response",
        )
    )
    worst = max(
        _ulp_error(float(a), float(b))
        for a, b in zip(torch_out.reshape(-1), jax_out.reshape(-1), strict=True)
    )
    assert worst <= 4.0, f"response parity worst_ulp={worst}"


def test_g3_argmax_within_4_ulp() -> None:
    import jax
    import jax.numpy as jnp
    from omnibias.jax.scan import bias_scan, init_bias_scan
    from omnibias.torch.scan import BiasScan

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    bank = BankSpec((-1.0, -0.5, 0.0, 0.5, 1.0))
    unit = BiasScan(
        1,
        bank,
        template="grad",
        base="tanh",
        learnable_offsets=False,
        learnable_scales=False,
        readout="argmax",
        gamma=8.0,
        dtype=torch.float64,
    )
    z_np = np.array([[0.3], [0.0], [-0.2]], dtype=np.float64)
    torch_out = unit(torch.as_tensor(z_np)).detach().numpy().reshape(-1)
    act, offsets, scales, tmpl, _taps = init_bias_scan(1, bank, template="grad", base="tanh")
    jax_out = np.asarray(
        bias_scan(
            jnp.asarray(z_np),
            offsets,
            scales,
            tmpl,
            act,
            readout="argmax",
            gamma=8.0,
        )
    ).reshape(-1)
    worst = max(_ulp_error(float(a), float(b)) for a, b in zip(torch_out, jax_out, strict=True))
    assert worst <= 4.0, f"argmax parity worst_ulp={worst}"


def test_unknown_template_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        from omnibias.torch.scan import template_from_op

        template_from_op("nope")
