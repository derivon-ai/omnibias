# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""torch / jax adaptive-refinement parity (theory 03-13 G6)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.core.refine import Indicator, RefinedPack, RefinePolicy  # noqa: E402
from omnibias.jax.refine import bank_forward, init_pack_bank  # noqa: E402
from omnibias.jax.refine import refine as jax_refine  # noqa: E402
from omnibias.torch.refine import AdaptivePackBank  # noqa: E402
from omnibias.torch.refine import refine as torch_refine  # noqa: E402

_BL_CENTER = 0.005
_BL_DERIVS = tuple((-100.0) ** k * math.exp(-0.5) for k in range(7))
_PACKS = (
    RefinedPack(order=0, center=0.25, weight=1.0, scale=2.0, age=0),
    RefinedPack(order=0, center=0.75, weight=0.5, scale=2.0, age=0),
)


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def test_forward_and_refine_decisions_match() -> None:
    torch.set_default_dtype(torch.float64)
    t_bank = AdaptivePackBank(_PACKS, max_packs=8, max_order=4, base="exp")
    j_bank = init_pack_bank(_PACKS, max_packs=8, max_order=4, base="exp")
    xs = np.linspace(0.0, 1.0, 33, dtype=np.float64)
    t_x = torch.as_tensor(xs)
    j_x = jnp.asarray(xs)
    t_y = [float(v) for v in t_bank(t_x).detach().cpu().reshape(-1)]
    j_y = [float(v) for v in bank_forward(j_bank, j_x)]
    worst = max(_ulp_error(a, b) for a, b in zip(t_y, j_y, strict=True))
    assert worst <= 4.0, f"forward parity worst_ulp={worst}"

    policy = RefinePolicy(
        indicator=Indicator.SINGULARITY,
        birth_threshold=0.1,
        death_threshold=0.0,
        min_age=10_000,
        hysteresis=1.0,
        min_scale_ratio=1.0,
    )

    def t_res(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-100.0 * x)

    def j_res(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-100.0 * x)

    t_report = torch_refine(
        t_bank, t_res, t_x, policy, step=0, probe_jet=(_BL_CENTER, _BL_DERIVS)
    )
    j_bank, j_report = jax_refine(
        j_bank, j_res, j_x, policy, step=0, probe_jet=(_BL_CENTER, _BL_DERIVS)
    )
    assert t_report.hp_move == j_report.hp_move
    assert t_report.proposed_center == j_report.proposed_center
    assert t_report.proposed_scale == j_report.proposed_scale
    assert len(t_report.born) == len(j_report.born)
    t_after = [float(v) for v in t_bank(t_x).detach().cpu().reshape(-1)]
    j_after = [float(v) for v in bank_forward(j_bank, j_x)]
    worst_after = max(_ulp_error(a, b) for a, b in zip(t_after, j_after, strict=True))
    assert worst_after <= 4.0, f"post-refine forward parity worst_ulp={worst_after}"
