# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch adaptive pack refinement (theory 03-13): G1/G2/G3 and GrowableOMBU."""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.core.refine import Indicator, RefinedPack, RefinePolicy, death_bound_holds
from omnibias.torch.growable import GrowableOperatorMultiBiasUnit
from omnibias.torch.refine import AdaptivePackBank, grow_ombu, refine, scalar_jet_1d

_BL_CENTER = 0.005
_BL_SCALE = -100.0
_BL_DERIVS = tuple((-100.0) ** k * math.exp(-0.5) for k in range(7))


@pytest.fixture(autouse=True)
def _f64() -> None:
    torch.set_default_dtype(torch.float64)


def _initial_bank(**kwargs: object) -> AdaptivePackBank:
    packs = (
        RefinedPack(order=0, center=0.25, weight=1.0, scale=2.0, age=0),
        RefinedPack(order=0, center=0.75, weight=0.5, scale=2.0, age=0),
    )
    return AdaptivePackBank(packs, max_packs=8, max_order=4, base="exp", **kwargs)  # type: ignore[arg-type]


def test_birth_bit_identical() -> None:
    bank = _initial_bank()
    probes = torch.linspace(0.0, 1.0, 33)

    def residual(x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    before = bank(probes).clone()
    report = refine(
        bank,
        residual,
        probes,
        RefinePolicy(
            indicator=Indicator.SINGULARITY,
            birth_threshold=0.1,
            death_threshold=0.0,
            min_age=10_000,
            hysteresis=1.0,
            min_scale_ratio=1.0,
            debug_zero_perturbation=True,
        ),
        step=0,
        probe_jet=(_BL_CENTER, _BL_DERIVS),
    )
    after = bank(probes)
    assert report.born
    assert torch.equal(before, after)
    assert report.born[0].weight == 0.0
    assert report.proposed_scale == pytest.approx(_BL_SCALE)


def test_growth_bit_identical() -> None:
    bank = AdaptivePackBank(
        (RefinedPack(order=0, center=0.0, weight=0.4, scale=-100.0, age=0),),
        max_packs=4,
        max_order=4,
        base="exp",
    )
    probes = torch.linspace(0.0, 0.2, 21)

    def residual(x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    before = bank(probes).clone()
    report = refine(
        bank,
        residual,
        probes,
        RefinePolicy(
            indicator=Indicator.SINGULARITY,
            birth_threshold=0.1,
            death_threshold=0.0,
            min_age=10_000,
            hysteresis=1.0,
            hp_rule="always_p",
            debug_zero_perturbation=True,
        ),
        step=0,
        probe_jet=(0.0, _BL_DERIVS),
    )
    assert report.grown == (0,)
    assert torch.equal(before, bank(probes))


def test_death_perturbation_bounds_change() -> None:
    bank = AdaptivePackBank(
        (
            RefinedPack(order=0, center=0.0, weight=1.0, scale=-2.0, age=200),
            RefinedPack(order=0, center=0.25, weight=0.003, scale=2.0, age=200),
        ),
        max_packs=4,
        base="exp",
    )
    probes = torch.linspace(0.0, 1.0, 33)
    before = bank(probes).clone()

    def residual(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    report = refine(
        bank,
        residual,
        probes,
        RefinePolicy(
            indicator=Indicator.RESIDUAL,
            birth_threshold=10.0,
            death_threshold=0.05,
            min_age=0,
            hysteresis=1.0,
        ),
        step=0,
    )
    assert report.died
    after = bank(probes)
    delta = float((after - before).square().sum().sqrt().item())
    field_norm = float(before.square().sum().sqrt().item())
    assert death_bound_holds(delta / field_norm, report.death_perturbation, atol=1e-12)


def test_residual_indicator_misses_scale() -> None:
    bank = _initial_bank()
    probes = torch.linspace(0.0, 1.0, 65)

    def residual(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-100.0 * x)

    report = refine(
        bank,
        residual,
        probes,
        RefinePolicy(
            indicator=Indicator.RESIDUAL,
            birth_threshold=0.1,
            death_threshold=0.0,
            min_age=10_000,
            hysteresis=1.0,
            min_scale_ratio=1.0,
        ),
        step=0,
    )
    assert report.proposed_scale is not None
    assert abs(report.proposed_scale) / 100.0 < 0.5


def test_g3_singularity_and_scale_flow() -> None:
    probes = torch.linspace(0.0, 1.0, 65)
    for kind in (Indicator.SINGULARITY, Indicator.SCALE_FLOW):
        bank = _initial_bank()

        def residual(x: torch.Tensor) -> torch.Tensor:
            return torch.exp(-100.0 * x)

        report = refine(
            bank,
            residual,
            probes,
            RefinePolicy(
                indicator=kind,
                birth_threshold=0.1,
                death_threshold=0.0,
                min_age=10_000,
                hysteresis=1.0,
                min_scale_ratio=1.0,
            ),
            step=0,
            probe_jet=(_BL_CENTER, _BL_DERIVS),
        )
        assert report.proposed_scale == pytest.approx(_BL_SCALE)
        assert 50.0 <= abs(float(report.proposed_scale or 0.0)) <= 200.0


def test_grow_ombu_reuses_existing_zero_perturbation() -> None:
    unit = GrowableOperatorMultiBiasUnit(num_channels=1, init_K=1, K_max=8)
    z = torch.linspace(-1.0, 1.0, 17).unsqueeze(-1)
    before = unit(z).clone()
    added = grow_ombu(unit, "pair")
    assert added == 2
    assert torch.equal(before, unit(z))


def test_scalar_jet_exp() -> None:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(3.0 * x)

    derivs = scalar_jet_1d(fn, 0.0, 4)
    for k, val in enumerate(derivs):
        assert val == pytest.approx(3.0**k, rel=1e-10)


def test_budget_stability_with_death() -> None:
    bank = _initial_bank()
    probes = torch.linspace(0.0, 1.0, 17)
    policy = RefinePolicy(
        indicator=Indicator.RESIDUAL,
        birth_threshold=0.5,
        death_threshold=0.2,
        min_age=2,
        hysteresis=1.0,
        max_packs=6,
        min_scale_ratio=1.0,
    )
    counts: list[int] = []
    for step in range(24):
        amp = 0.8 if step % 4 == 0 else 0.0

        def residual(x: torch.Tensor, _amp: float = amp) -> torch.Tensor:
            return torch.full_like(x, _amp)

        refine(bank, residual, probes, policy, step=step)
        counts.append(bank.n_active)
    assert max(counts) <= 6
    assert counts[-1] <= counts[0] + 2
