# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Reusable PirateNet α-skip (torch twin of :mod:`omnibias.jax.architectures.piratenet`).

Identity-init residual blocks: ``α=0`` is the embedding. Not ImageNet / ViT,
not CCF stretch, not a Wave-3 gated invention. Apply is bit-identical to the
JAX twin given the same weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class PirateNetConfig:
    """PirateNet width / depth. ``α`` starts at 0."""

    in_dim: int = 1
    hidden: int = 24
    n_layers: int = 2
    out_dim: int = 1
    seed: int = 0


def _glorot(shape: tuple[int, ...], generator: torch.Generator) -> Tensor:
    fan_in = int(shape[-1]) if len(shape) > 1 else 1
    scale = (2.0 / float(fan_in)) ** 0.5
    noise = torch.randn(shape, generator=generator, dtype=torch.float64)
    return torch.as_tensor(scale * noise, dtype=torch.float64)


def init_pirate_params(cfg: PirateNetConfig) -> dict[str, Any]:
    """Identity-skip PirateNet (``alpha=0``) plus a zero readout."""
    hid = int(cfg.hidden)
    n_layers = int(cfg.n_layers)
    in_dim = int(cfg.in_dim)
    g = torch.Generator().manual_seed(int(cfg.seed))
    if int(cfg.out_dim) == 1:
        wout: Tensor = torch.zeros(hid, dtype=torch.float64)
        bout: Tensor = torch.zeros((), dtype=torch.float64)
    else:
        wout = torch.zeros(hid, int(cfg.out_dim), dtype=torch.float64)
        bout = torch.zeros(int(cfg.out_dim), dtype=torch.float64)
    params: dict[str, Any] = {
        "We": _glorot((hid, in_dim), g),
        "be": torch.zeros(hid, dtype=torch.float64),
        "Wu": _glorot((hid, hid), g),
        "bu": torch.zeros(hid, dtype=torch.float64),
        "Wv": _glorot((hid, hid), g),
        "bv": torch.zeros(hid, dtype=torch.float64),
        "Wout": wout,
        "bout": bout,
        "alpha": torch.zeros(n_layers, dtype=torch.float64),
        "blocks": [],
    }
    for _ in range(n_layers):
        params["blocks"].append(
            {
                "W1": _glorot((hid, hid), g),
                "b1": torch.zeros(hid, dtype=torch.float64),
                "W2": _glorot((hid, hid), g),
                "b2": torch.zeros(hid, dtype=torch.float64),
                "W3": _glorot((hid, hid), g),
                "b3": torch.zeros(hid, dtype=torch.float64),
            }
        )
    params["blocks"] = tuple(params["blocks"])
    return params


def pirate_features(params: dict[str, Any], x: Tensor) -> Tensor:
    """Penultimate features, shape ``(..., hidden)``. ``alpha=0`` is identity."""
    coords = torch.as_tensor(x, dtype=torch.float64)
    if coords.ndim == 1:
        coords = coords[None, :]
        squeeze = True
    else:
        squeeze = False
    h = torch.tanh(coords @ params["We"].T + params["be"])
    u = torch.tanh(h @ params["Wu"].T + params["bu"])
    v = torch.tanh(h @ params["Wv"].T + params["bv"])
    for i, block in enumerate(params["blocks"]):
        identity = h
        z = torch.tanh(h @ block["W1"].T + block["b1"])
        z = z * u + (1.0 - z) * v
        z = torch.tanh(z @ block["W2"].T + block["b2"])
        z = z * u + (1.0 - z) * v
        z = torch.tanh(z @ block["W3"].T + block["b3"])
        alpha = params["alpha"][i]
        h = alpha * z + (1.0 - alpha) * identity
    if squeeze:
        return torch.as_tensor(h[0])
    return torch.as_tensor(h)


def pirate_apply(params: dict[str, Any], x: Tensor) -> Tensor:
    """Linear readout of :func:`pirate_features`."""
    feat = pirate_features(params, x)
    return torch.as_tensor(feat @ params["Wout"] + params["bout"])


class PirateNet(nn.Module):
    """Trainable PirateNet wrapping the functional apply."""

    def __init__(self, cfg: PirateNetConfig | None = None) -> None:
        super().__init__()
        cfg = PirateNetConfig() if cfg is None else cfg
        raw = init_pirate_params(cfg)
        self.in_dim = int(cfg.in_dim)
        self.hidden = int(cfg.hidden)
        self.n_layers = int(cfg.n_layers)
        self.out_dim = int(cfg.out_dim)
        self.We = nn.Parameter(raw["We"])
        self.be = nn.Parameter(raw["be"])
        self.Wu = nn.Parameter(raw["Wu"])
        self.bu = nn.Parameter(raw["bu"])
        self.Wv = nn.Parameter(raw["Wv"])
        self.bv = nn.Parameter(raw["bv"])
        self.Wout = nn.Parameter(raw["Wout"])
        self.bout = nn.Parameter(raw["bout"])
        self.alpha = nn.Parameter(raw["alpha"])
        blocks = []
        for block in raw["blocks"]:
            blocks.append(
                nn.ParameterDict(
                    {
                        "W1": nn.Parameter(block["W1"]),
                        "b1": nn.Parameter(block["b1"]),
                        "W2": nn.Parameter(block["W2"]),
                        "b2": nn.Parameter(block["b2"]),
                        "W3": nn.Parameter(block["W3"]),
                        "b3": nn.Parameter(block["b3"]),
                    }
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def _params(self) -> dict[str, Any]:
        return {
            "We": self.We,
            "be": self.be,
            "Wu": self.Wu,
            "bu": self.bu,
            "Wv": self.Wv,
            "bv": self.bv,
            "Wout": self.Wout,
            "bout": self.bout,
            "alpha": self.alpha,
            "blocks": [
                {
                    k: cast(nn.ParameterDict, block)[k]
                    for k in ("W1", "b1", "W2", "b2", "W3", "b3")
                }
                for block in self.blocks
            ],
        }

    def forward(self, x: Tensor) -> Tensor:
        return pirate_apply(self._params(), x)


__all__ = [
    "PirateNet",
    "PirateNetConfig",
    "init_pirate_params",
    "pirate_apply",
    "pirate_features",
]
