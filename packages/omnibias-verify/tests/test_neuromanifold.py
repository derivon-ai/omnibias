# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
import torch
from omnibias.core.confluence import cluster_remainder, initialize_moments, pair_series_error
from omnibias.core.realization.transition import SlotUpdate, TransitionProposal
from omnibias.core.refine import RefinedPack
from omnibias.core.verified.interval import Interval
from omnibias.geometry.neuromanifold import affine_quotient
from omnibias.torch.confluent_bank import ConfluentPackBank, commit_transition
from omnibias.torch.refine import AdaptivePackBank
from omnibias.verify._core.param_loss import HyperDual
from omnibias.verify.neuromanifold import (
    IntervalObjective,
    certify_pair_collapse,
    certify_quotient_minimum,
    certify_slice_minimum,
    certify_transition,
    integral_error,
    linear_residual_error,
)


def test_whole_box_minimum_encloses_actual_root_and_quotient():
    def objective(q):
        a, b = q[0] - HyperDual.constant(3), q[1] + HyperDual.constant(Fraction(1, 5))
        return a * a + b * b

    obj = IntervalObjective(objective, "quadratic_shifted", ("q0", "q1"))
    box = (Interval(2.9, 3.2), Interval(-0.3, -0.1))
    result = certify_slice_minimum(obj, box)
    assert result.status == "proved" and result.claim == "slice_minimum"
    assert result.stationary_box[0].contains(3) and result.stationary_box[1].contains(-0.2)
    assert result.stationary_box[0].width < 0.001
    chart = affine_quotient([[1, 1, 0], [0, 0, 1]])
    quotient = certify_quotient_minimum(chart, obj, box, morse_bott=True)
    assert quotient.status == "proved" and quotient.claim == "morse_bott_minimum"
    wrong = replace(
        chart, projection=((Fraction(1), Fraction(0), Fraction(0)), chart.projection[1])
    )
    with pytest.raises(ValueError, match="invalid"):
        certify_quotient_minimum(wrong, obj, box)
    saddle = IntervalObjective(lambda q: q[0] * q[0] - q[1] * q[1], "saddle", ("a", "b"))
    assert certify_slice_minimum(saddle, (Interval(-0.1, 0.1),) * 2).status == "inconclusive"


def test_pair_derivative_and_fourth_order_residual_enclosures():
    domain = Interval(-1, 1)
    result = certify_pair_collapse(
        domain=domain,
        rho=Interval.point(1e-10),
        m0=Interval.point(0.7),
        m1=Interval.point(-0.9),
        weight=Interval.point(0.8),
        bias=Interval.point(0.2),
        direction=Interval.point(-0.4),
        direction_bias=Interval.point(0.6),
        spatial_orders=(0, 1, 2, 3, 4),
        error_budget=0.1,
    )
    assert result.accepted
    rng = np.random.default_rng(2026)
    grid = np.r_[np.linspace(-1, 1, 201), rng.uniform(-1, 1, 100)]
    x = torch.tensor(grid, dtype=torch.float64, requires_grad=True)
    z, eta, h = 0.8 * x + 0.2, -0.4 * x + 0.6, 1e-5
    pair = 0.7 * (torch.sigmoid(z + h * eta) + torch.sigmoid(z - h * eta)) / 2 - 0.9 * (
        torch.sigmoid(z + h * eta) - torch.sigmoid(z - h * eta)
    ) / (2 * h)
    s = torch.sigmoid(z)
    target = 0.7 * s - 0.9 * eta * s * (1 - s)
    errors = []
    for n, enclosure in result.errors:
        if n:
            pair = torch.autograd.grad(pair.sum(), x, create_graph=True)[0]
            target = torch.autograd.grad(target.sum(), x, create_graph=True)[0]
        assert float((pair - target).abs().max().detach()) <= enclosure.hi
        errors.append(enclosure)
    # A fourth-order constant-coefficient residual, composed from the same jets.
    residual_error = linear_residual_error([1.0, 0.0, -2.0, 0.0, 1.0], errors)
    assert residual_error.hi < 0.1
    window = integral_error(errors[0], Interval.point(2), kind="activation_window")
    assert window.hi >= 2 * errors[0].hi
    assert certify_pair_collapse(
        domain=domain,
        rho=Interval.point(0),
        m0=Interval.point(1),
        m1=Interval.point(2),
        error_budget=0,
    ).accepted


def test_cluster_moments_exact_conversion_and_remainder():
    initialization = initialize_moments((-0.001, 0.002, 0.003), (1.2, -0.7, 0.3), order=4)
    for q, f, error in zip(
        initialization.exact_moments,
        initialization.moments,
        initialization.conversion_errors,
        strict=True,
    ):
        assert error.contains(float(q - Fraction(f)))
    x = torch.tensor(
        np.r_[np.linspace(-2, 2, 201), np.random.default_rng(42).uniform(-2, 2, 100)],
        dtype=torch.float64,
        requires_grad=True,
    )
    actual = sum(
        a * torch.sigmoid(x + b)
        for a, b in zip((1.2, -0.7, 0.3), (-0.001, 0.002, 0.003), strict=True)
    )
    derivatives = [torch.sigmoid(x)]
    for _ in range(4):
        derivatives.append(torch.autograd.grad(derivatives[-1].sum(), x, create_graph=True)[0])
    approximation = sum(m * d for m, d in zip(initialization.moments, derivatives, strict=True))
    # Add a small independent floating oracle tolerance; the enclosure concerns
    # the real analytic expansion, not rounding of this sampling calculation.
    assert (
        float((actual - approximation).abs().max().detach())
        < cluster_remainder(initialization).hi + 1e-15
    )
    a, b = pair_series_error(rho=Interval.point(1e-4), eta=Interval.point(1), terms=6)
    assert a.hi > 0 and b.hi > 0


def _bank():
    return ConfluentPackBank(
        [
            RefinedPack(center=-0.03, weight=0.7, scale=1.0, order=0),
            RefinedPack(center=0.03, weight=-0.2, scale=1.0, order=0),
            RefinedPack(center=0.4, weight=0.1, scale=1.0, order=0),
        ],
        max_packs=4,
        max_order=4,
        dtype=torch.float64,
    )


def test_atomic_transition_staleness_rollback_state_and_checkpoint():
    bank = _bank()
    optimizer = torch.optim.Adam(bank.parameters(), lr=1e-3)
    grid = torch.linspace(-1, 1, 11, dtype=torch.float64)
    bank(grid).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # The test exercises common-weight grouping, so restore the common scales
    # after independent optimizer updates before producing the proposal.
    with torch.no_grad():
        bank.scales.fill_(1)
    source = bank(grid).detach().clone()
    ids = [id(p) for p in bank.parameters()]
    untouched = optimizer.state[bank.weights]["exp_avg"][2].clone()
    proposal = bank.propose_pair(0, 1, error_budget=1e-10)
    assert TransitionProposal.from_json(proposal.to_json()) == proposal

    def acceptance(b, p):
        return certify_transition(b.portable_snapshot(), p, domain=Interval(-1, 1)).accepted

    assert not commit_transition(
        bank,
        proposal,
        optimizer,
        between_steps=True,
        acceptance=acceptance,
        postcondition=lambda _: False,
    )
    torch.testing.assert_close(bank(grid), source, rtol=0, atol=0)
    assert commit_transition(bank, proposal, optimizer, between_steps=True, acceptance=acceptance)
    assert [id(p) for p in bank.parameters()] == ids
    assert optimizer.state[bank.weights]["exp_avg"][0] == 0
    assert optimizer.state[bank.weights]["exp_avg"][1] == 0
    assert optimizer.state[bank.weights]["exp_avg"][2] == untouched
    torch.testing.assert_close(bank(grid), source, rtol=1e-10, atol=1e-12)
    with pytest.raises(ValueError, match="stale"):
        commit_transition(bank, proposal, optimizer, between_steps=True, acceptance=acceptance)
    restored = _bank()
    restored.load_state_dict(bank.state_dict())
    torch.testing.assert_close(restored(grid), bank(grid), rtol=0, atol=0)
    # Source operands cannot be swapped after the proposal is serialized/resealed.
    bad = replace(proposal, updates=proposal.updates + (SlotUpdate("weights", 2, 100.0),))
    with pytest.raises(ValueError):
        certify_transition(_bank().portable_snapshot(), bad, domain=Interval(-1, 1))


def test_legacy_migration_and_unknown_optimizer_refusal():
    old = AdaptivePackBank(
        [RefinedPack(center=0, weight=1, scale=1, order=0)],
        max_packs=4,
        max_order=4,
        base="sigmoid",
        dtype=torch.float64,
    )
    new = _bank()
    new.load_state_dict(old.state_dict())
    x = torch.tensor([-0.2, 0.3], dtype=torch.float64)
    torch.testing.assert_close(old(x), new(x), rtol=0, atol=0)
    assert not new.modes.any()

    class Unknown(torch.optim.SGD):
        pass

    bank = _bank()
    optimizer = Unknown(bank.parameters(), lr=0.01)
    proposal = bank.propose_pair(0, 1, error_budget=1e-8)
    digest = bank.portable_snapshot()
    with pytest.raises(TypeError, match="adapter"):
        commit_transition(bank, proposal, optimizer, between_steps=True, acceptance=lambda *_: True)
    assert bank.portable_snapshot() == digest


def test_cluster_and_birth_transaction_certificates():
    from omnibias.verify.neuromanifold import replay_confluence_certificate

    bank = _bank()
    with torch.no_grad():
        bank.centers[:3].copy_(torch.tensor([-0.001, 0.002, 0.003], dtype=torch.float64))
    optimizer = torch.optim.Adam(bank.parameters(), lr=0.01)
    proposal = bank.propose_cluster((0, 1, 2), order=4, error_budget=1e-5)
    result = certify_transition(
        bank.portable_snapshot(), proposal, domain=Interval(-1, 1), spatial_orders=(0, 4)
    )
    assert result.accepted and replay_confluence_certificate(result.certificate)
    assert commit_transition(
        bank, proposal, optimizer, between_steps=True, acceptance=lambda *_: result.accepted
    )
    birth = bank.propose_birth(1, RefinedPack(center=0.5, weight=0, scale=0.7, order=2))
    result = certify_transition(
        bank.portable_snapshot(), birth, domain=Interval(-1, 1), spatial_orders=(0, 4)
    )
    assert result.accepted and result.budget == 0
    assert commit_transition(
        bank, birth, optimizer, between_steps=True, acceptance=lambda *_: result.accepted
    )


def test_runtime_series_is_in_budget_and_switch_crossing_is_inconclusive():
    bank = _bank()
    with torch.no_grad():
        bank.centers[:2].copy_(torch.tensor([-0.005, 0.005], dtype=torch.float64))
    proposal = bank.propose_pair(0, 1, error_budget=0)
    result = certify_transition(bank.portable_snapshot(), proposal, domain=Interval(-1, 1))
    assert not result.accepted and result.certificate["meta"]["runtime_series_included"]
    with torch.no_grad():
        bank.modes[0] = 1
        bank.rho[0] = 0.0001
        bank.directions[0] = 1
        bank.direction_bias[0] = 1
    proposal = bank.propose_derivative(0, error_budget=100)
    result = certify_transition(
        bank.portable_snapshot(), proposal, domain=Interval(-1, 1), spatial_orders=(1,)
    )
    assert not result.accepted and result.certificate["meta"]["spatial_series_switch_crossing"]


def test_finite_architecture_temperature_gap_and_gradient():
    import jax
    from omnibias.verify.neuromanifold.selection import (
        select_architectures_jax,
        select_architectures_torch,
    )

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    costs = torch.tensor([1.3, 0.7, 2.1], dtype=torch.float64, requires_grad=True)
    result = select_architectures_torch(costs, ("a", "b", "c"), beta=4)
    hard = float(costs.detach().min())
    soft = float(result.soft_minimum.detach())
    assert soft <= hard <= soft + result.gap.hi
    torch.testing.assert_close(torch.autograd.grad(result.soft_minimum, costs)[0], result.weights)
    twin = select_architectures_jax(jnp.array([1.3, 0.7, 2.1]), ("a", "b", "c"), beta=4)
    np.testing.assert_allclose(result.weights.detach(), twin.weights, rtol=1e-12, atol=1e-12)


def test_minimum_finite_replay_and_unavailable_toolchain(monkeypatch):
    from omnibias.core.proof.realization_replay import verify_replay_certificate
    from omnibias.verify.neuromanifold import formalize_minimum, minimum_replay_certificates

    obj = IntervalObjective(lambda q: q[0] * q[0], "square", ("x",))
    result = certify_slice_minimum(obj, (Interval(-0.1, 0.1),))
    witnesses = minimum_replay_certificates(result)
    assert all(verify_replay_certificate(w) for w in witnesses)
    monkeypatch.setenv("PATH", "")
    formal = formalize_minimum(result)
    assert not formal.theorem_prover_verified and not formal.kernel[0].available


def test_attained_zero_spread_transition_preserves_exact_zero_budget():
    bank = _bank()
    proposal = bank.propose_pair(0, 1, error_budget=1e-8)
    optimizer = torch.optim.Adam(bank.parameters())
    assert commit_transition(
        bank, proposal, optimizer, between_steps=True, acceptance=lambda _bank, _proposal: True
    )
    with torch.no_grad():
        bank.rho[0] = 0
    derivative = bank.propose_derivative(0, error_budget=0)
    result = certify_transition(
        bank.portable_snapshot(), derivative, domain=Interval(-1, 1), spatial_orders=(0, 1, 2, 3, 4)
    )
    assert result.accepted
    assert all(error.lo == error.hi == 0 for _, error in result.errors)
