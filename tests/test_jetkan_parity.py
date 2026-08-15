# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-KAN torch/jax parity (theory 02-03 G5)."""

from __future__ import annotations

import math

import numpy as np
import torch
from omnibias.torch.architectures.jetkan import JetKAN, JetKANConfig


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def _torch_state(net: JetKAN) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    for layer in net.layers:
        out.append(
            (
                layer.weights.detach().cpu().numpy(),
                layer.means.detach().cpu().numpy(),
                layer.log_scales.detach().cpu().numpy(),
                int(layer.active_g),
            )
        )
    return out


def test_jetkan_torch_jax_parity() -> None:
    import jax
    import jax.numpy as jnp
    from omnibias.jax.architectures.jetkan import (
        JetKANConfig as JaxCfg,
    )
    from omnibias.jax.architectures.jetkan import (
        jet_kan_apply,
        jet_kan_from_torch_state,
    )

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    cfg = JetKANConfig(widths=(2, 2, 1), packs_per_edge=2, extra_packs=1, orders=(0, 1))
    net = JetKAN(cfg, dtype=torch.float64)
    x_np = np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.0]], dtype=np.float64)
    t_out = net(torch.as_tensor(x_np)).detach().numpy()
    jcfg = JaxCfg(
        widths=cfg.widths,
        packs_per_edge=cfg.packs_per_edge,
        extra_packs=cfg.extra_packs,
        orders=cfg.orders,
        base=cfg.base,
    )
    params = jet_kan_from_torch_state(jcfg, _torch_state(net))
    j_out = np.asarray(jet_kan_apply(params, jnp.asarray(x_np), config=jcfg))
    worst = max(
        _ulp_error(float(a), float(b))
        for a, b in zip(t_out.reshape(-1), j_out.reshape(-1), strict=True)
    )
    assert worst <= 4.0, f"JetKAN parity worst_ulp={worst}"
