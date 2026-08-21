# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Adaptive pack-refinement algebra (theory 03-13): worked BL jet, death bound."""

from __future__ import annotations

import math

import pytest
from omnibias.core.multipack import PackSpec
from omnibias.core.refine import (
    Indicator,
    RefinedPack,
    RefinePolicy,
    RefineProposal,
    apply_birth,
    apply_deaths,
    apply_growth,
    assert_zero_perturbation,
    certified_local_error,
    death_bound_holds,
    domb_sykes_fit,
    hp_decision,
    increment_ages,
    inherit_scale,
    local_scale_from_derivatives,
    pack_contributions,
    pack_term_scalar,
    propose_refinement,
    residual_indicator,
    select_deaths,
)

# Spec §5: Taylor of exp(-100 x) about x=0.005 is
# a_k = (-100)^k exp(-0.5) / k!, so |a_k / a_{k-1}| = 100 / k.
_BL_CENTER = 0.005
_BL_SCALE = -100.0
_BL_DERIVS = tuple((-100.0) ** k * math.exp(-0.5) for k in range(7))


def _exp_sigma(u: float, n: int) -> float:
    return math.exp(u)


def test_boundary_layer_local_scale() -> None:
    scale = local_scale_from_derivatives(_BL_DERIVS)
    assert scale == pytest.approx(_BL_SCALE, rel=1e-12)


def test_boundary_layer_domb_sykes_entire() -> None:
    intercept, slope = domb_sykes_fit(_BL_DERIVS)
    assert intercept == pytest.approx(0.0, abs=1e-12)
    assert slope == pytest.approx(_BL_SCALE, rel=1e-12)


def test_boundary_layer_hp_is_h_then_p() -> None:
    assert hp_decision(_BL_DERIVS, existing_scale=2.0) == "h"
    assert hp_decision(_BL_DERIVS, existing_scale=-100.0) == "p"


def test_residual_indicator_inherits_wrong_scale() -> None:
    packs = (
        RefinedPack(order=0, center=0.25, weight=1.0, scale=2.0),
        RefinedPack(order=0, center=0.75, weight=1.0, scale=2.0),
    )
    policy = RefinePolicy(
        indicator=Indicator.RESIDUAL,
        birth_threshold=0.1,
        death_threshold=0.01,
        hysteresis=1.0,
        min_scale_ratio=1.0,
    )
    proposal = propose_refinement(
        packs,
        policy=policy,
        peak_location=_BL_CENTER,
        peak_value=1.0,
        derivatives=_BL_DERIVS,
        birth_score=1.0,
    )
    assert proposal is not None
    assert proposal.move == "h"
    assert proposal.inherit_scale
    assert abs(proposal.scale) == pytest.approx(2.0)
    assert abs(proposal.scale) / 100.0 < 0.5


def test_singularity_and_scale_flow_place_correct_scale() -> None:
    packs = (
        RefinedPack(order=0, center=0.25, weight=1.0, scale=2.0),
        RefinedPack(order=0, center=0.75, weight=1.0, scale=2.0),
    )
    for kind in (Indicator.SINGULARITY, Indicator.SCALE_FLOW):
        policy = RefinePolicy(
            indicator=kind,
            birth_threshold=0.1,
            death_threshold=0.01,
            hysteresis=1.0,
            min_scale_ratio=1.0,
        )
        proposal = propose_refinement(
            packs,
            policy=policy,
            peak_location=_BL_CENTER,
            peak_value=1.0,
            derivatives=_BL_DERIVS,
            birth_score=1.0,
        )
        assert proposal is not None
        assert proposal.move == "h"
        assert not proposal.inherit_scale
        assert proposal.scale == pytest.approx(_BL_SCALE, rel=1e-12)
        assert 50.0 <= abs(proposal.scale) <= 200.0


def test_birth_is_zero_weight() -> None:
    packs = (RefinedPack(order=0, center=0.25, weight=1.0, scale=2.0),)
    proposal = RefineProposal(move="h", center=_BL_CENTER, scale=_BL_SCALE, order=0)
    after, born = apply_birth(packs, proposal)
    assert born.weight == 0.0
    xs = [0.0, 0.01, 0.5, 1.0]
    before_vals = [sum(pack_term_scalar(x, p, _exp_sigma) for p in packs) for x in xs]
    after_vals = [sum(pack_term_scalar(x, p, _exp_sigma) for p in after) for x in xs]
    assert_zero_perturbation(before_vals, after_vals)


def test_growth_sibling_is_zero_perturbation() -> None:
    packs = (RefinedPack(order=0, center=0.0, weight=0.3, scale=-100.0),)
    proposal = RefineProposal(
        move="p",
        center=0.0,
        scale=-100.0,
        order=1,
        parent_index=0,
    )
    after, grown, parent = apply_growth(packs, proposal)
    assert parent == 0
    assert grown.weight == 0.0
    assert grown.order == 1
    xs = [0.0, 0.01, 0.25]
    before_vals = [sum(pack_term_scalar(x, p, _exp_sigma) for p in packs) for x in xs]
    after_vals = [sum(pack_term_scalar(x, p, _exp_sigma) for p in after) for x in xs]
    assert_zero_perturbation(before_vals, after_vals)


def test_death_perturbation_bounds_measured_change() -> None:
    packs = (
        RefinedPack(order=0, center=0.0, weight=1.0, scale=-2.0, age=200),
        RefinedPack(order=0, center=0.25, weight=0.003, scale=2.0, age=200),
    )
    xs = [i / 32.0 for i in range(33)]
    terms = []
    field_sq = 0.0
    for pack in packs:
        values = [pack_term_scalar(x, pack, _exp_sigma) for x in xs]
        terms.append(math.sqrt(sum(v * v for v in values)))
        field_sq += sum(v * v for v in values)
    field_norm = math.sqrt(field_sq)
    contrib = pack_contributions(terms, field_norm)
    died = select_deaths(packs, contrib, RefinePolicy(min_age=100, death_threshold=0.05))
    assert 1 in died
    kept, reported = apply_deaths(packs, died, contrib)
    assert len(kept) == 1
    before = [sum(pack_term_scalar(x, p, _exp_sigma) for p in packs) for x in xs]
    after = [sum(pack_term_scalar(x, p, _exp_sigma) for p in kept) for x in xs]
    delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(after, before, strict=True)))
    measured = delta / field_norm
    assert death_bound_holds(measured, reported, atol=1e-15)


def test_min_age_protects_newborn() -> None:
    packs = (
        RefinedPack(order=0, center=0.0, weight=1.0, scale=1.0, age=200),
        RefinedPack(order=0, center=0.1, weight=0.0, scale=1.0, age=0),
    )
    contrib = [0.9, 0.0]
    died = select_deaths(packs, contrib, RefinePolicy(min_age=100, death_threshold=0.1))
    assert died == []


def test_hysteresis_blocks_weak_birth() -> None:
    packs = (RefinedPack(order=0, center=0.5, weight=1.0, scale=2.0),)
    policy = RefinePolicy(
        indicator=Indicator.RESIDUAL,
        birth_threshold=0.1,
        death_threshold=0.05,
        hysteresis=4.0,
    )
    # score 0.15 exceeds birth_threshold but not hysteresis * death = 0.2
    proposal = propose_refinement(
        packs,
        policy=policy,
        peak_location=0.0,
        peak_value=0.15,
        derivatives=None,
        birth_score=0.15,
    )
    assert proposal is None


def test_residual_argmax() -> None:
    idx, mag = residual_indicator((0.1, -0.4, 0.2))
    assert idx == 1
    assert mag == pytest.approx(0.4)


def test_certified_local_error_exp() -> None:
    # exp on [0, 0.1]; order-3 model, M = |phi^(4)(0)| = 1 under-bounds
    # the true max e^{0.1}; still a well-defined remainder number.
    derivs = tuple(1.0 for _ in range(5))
    bound = certified_local_error(derivs, 0.1)
    remainder = abs(math.exp(0.1) - sum((0.1**k) / math.factorial(k) for k in range(4)))
    # Using M=1 < e^{0.1} is not sound; the API is only as sound as M.
    # With the true max M the bound must cover the remainder:
    sound = certified_local_error((1.0, 1.0, 1.0, 1.0, math.exp(0.1)), 0.1)
    assert sound >= remainder - 1e-15
    assert bound > 0.0


def test_pack_spec_roundtrip() -> None:
    spec = PackSpec(order=2, mean=-0.25, weight=0.5)
    pack = RefinedPack.from_pack_spec(spec, scale=3.0)
    assert pack.center == pytest.approx(0.25)
    assert pack.to_pack_spec().mean == pytest.approx(spec.mean)
    assert pack.to_pack_spec().order == spec.order


def test_increment_ages() -> None:
    packs = increment_ages((RefinedPack(order=0, center=0.0, weight=1.0, age=3),), 2)
    assert packs[0].age == 5


def test_inherit_scale_empty() -> None:
    assert inherit_scale((), 0.0, default=4.0) == 4.0
