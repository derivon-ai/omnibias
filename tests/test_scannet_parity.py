# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Scan-Net torch/jax parity (theory 02-01 G5)."""

from __future__ import annotations

import math

import numpy as np
import torch
from omnibias.torch.architectures.scannet import ScanNet, ScanNetConfig


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def _torch_state(net: ScanNet) -> list[tuple[np.ndarray, ...]]:
    out: list[tuple[np.ndarray, ...]] = []
    for layer in net.layers:
        out.append(
            (
                layer.proj.weight.detach().cpu().numpy(),
                layer.proj.bias.detach().cpu().numpy(),
                layer.scan.offsets.detach().cpu().numpy(),
                layer.scan.scales.detach().cpu().numpy(),
                layer.taps.detach().cpu().numpy(),
            )
        )
    return out


def test_scannet_torch_jax_pooled_parity() -> None:
    import jax
    import jax.numpy as jnp
    from omnibias.jax.architectures.scannet import (
        ScanNetConfig as JaxCfg,
    )
    from omnibias.jax.architectures.scannet import (
        scan_net_apply,
        scan_net_from_torch_state,
    )

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    cfg = ScanNetConfig(
        dim_in=2,
        channels=(3, 2),
        bank_sizes=(5, 7),
        bank_extents=(1.0, 1.5),
        template="grad",
        base="tanh",
        readout="pooled",
    )
    net = ScanNet(cfg, dtype=torch.float64)
    x_np = np.array([[0.0, 0.0], [0.3, -0.2], [-0.4, 0.5]], dtype=np.float64)
    torch_out = net(torch.as_tensor(x_np)).detach().numpy()
    jcfg = JaxCfg(
        dim_in=cfg.dim_in,
        channels=cfg.channels,
        bank_sizes=cfg.bank_sizes,
        bank_extents=cfg.bank_extents,
        template=cfg.template,
        base=cfg.base,
        readout=cfg.readout,
    )
    params = scan_net_from_torch_state(jcfg, _torch_state(net))
    jax_out = np.asarray(scan_net_apply(params, jnp.asarray(x_np), config=jcfg))
    worst = max(
        _ulp_error(float(a), float(b))
        for a, b in zip(torch_out.reshape(-1), jax_out.reshape(-1), strict=True)
    )
    assert worst <= 4.0, f"ScanNet parity worst_ulp={worst}"
