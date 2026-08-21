# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX adaptive pack refinement (theory 03-13): G1/G2/G3 twins."""

from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax import Array  # noqa: E402
from omnibias.core.refine import (  # noqa: E402
    Indicator,
    RefinedPack,
    RefinePolicy,
    death_bound_holds,
)
from omnibias.jax.refine import (  # noqa: E402
    bank_forward,
    init_pack_bank,
    refine,
    scalar_jet_1d,
)

_BL_CENTER = 0.005
_BL_SCALE = -100.0
_BL_DERIVS = tuple((-100.0) ** k * math.exp(-0.5) for k in range(7))


def _initial_bank() -> object:
    packs = (
        RefinedPack(order=0, center=0.25, weight=1.0, scale=2.0, age=0),
        RefinedPack(order=0, center=0.75, weight=0.5, scale=2.0, age=0),
    )
    return init_pack_bank(packs, max_packs=8, max_order=4, base="exp")


def test_birth_bit_identical() -> None:
    bank = _initial_bank()
    probes = jnp.linspace(0.0, 1.0, 33)

    def residual(x: Array) -> Array:
        return jnp.ones_like(x)

    before = bank_forward(bank, probes)
    bank, report = refine(
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
    after = bank_forward(bank, probes)
    assert report.born
    assert bool(jnp.array_equal(before, after))
    assert report.proposed_scale == pytest.approx(_BL_SCALE)


def test_death_perturbation_bounds_change() -> None:
    bank = init_pack_bank(
        (
            RefinedPack(order=0, center=0.0, weight=1.0, scale=-2.0, age=200),
            RefinedPack(order=0, center=0.25, weight=0.003, scale=2.0, age=200),
        ),
        max_packs=4,
        base="exp",
    )
    probes = jnp.linspace(0.0, 1.0, 33)
    before = bank_forward(bank, probes)

    def residual(x: Array) -> Array:
        return jnp.zeros_like(x)

    bank, report = refine(
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
    after = bank_forward(bank, probes)
    delta = float(jnp.sqrt(jnp.sum((after - before) ** 2)))
    field_norm = float(jnp.sqrt(jnp.sum(before**2)))
    assert death_bound_holds(delta / field_norm, report.death_perturbation, atol=1e-12)


def test_g3_scale_flow() -> None:
    bank = _initial_bank()
    probes = jnp.linspace(0.0, 1.0, 65)

    def residual(x: Array) -> Array:
        return jnp.exp(-100.0 * x)

    _bank, report = refine(
        bank,
        residual,
        probes,
        RefinePolicy(
            indicator=Indicator.SCALE_FLOW,
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


def test_scalar_jet_exp() -> None:
    def fn(x: Array) -> Array:
        return jnp.exp(3.0 * x)

    derivs = scalar_jet_1d(fn, 0.0, 4)
    for k, val in enumerate(derivs):
        assert val == pytest.approx(3.0**k, rel=1e-10)
