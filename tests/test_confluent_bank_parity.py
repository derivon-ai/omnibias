# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Portable collision transactions, functional optimizer state and live JIT."""

from copy import deepcopy
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.realization.transition import TransitionProposal
from omnibias.core.refine import RefinedPack
from omnibias.jax import confluent_bank as jb
from omnibias.jax.refine import bank_forward as legacy_forward
from omnibias.jax.refine import init_pack_bank
from omnibias.torch import confluent_bank as tb
from omnibias.torch.refine import AdaptivePackBank

jax.config.update("jax_enable_x64", True)


def pair_banks():
    packs = [RefinedPack(0, -0.125, 2.0), RefinedPack(0, 0.125, -1.0), RefinedPack(1, 0.2, 0.3)]
    return (tb.ConfluentPackBank(packs, max_packs=4, max_order=3, dtype=torch.float64),
            jb.init_confluent_bank(packs, max_packs=4, max_order=3, dtype=jnp.float64))


def moments(bank):
    return jb.NamedMomentState(jnp.asarray(7), {
        "first": {name: jnp.ones_like(p) for name, p in bank.parameters.items()},
        "second": {name: 2 * jnp.ones_like(p) for name, p in bank.parameters.items()},
    })


def test_portable_proposals_and_functional_transaction_match_torch() -> None:
    torch_bank, bank = pair_banks()
    assert bank.portable_snapshot() == torch_bank.portable_snapshot()
    proposal = bank.propose_pair(0, 1)
    assert proposal == torch_bank.propose_pair(0, 1)
    assert TransitionProposal.from_json(proposal.to_json()) == proposal
    state = moments(bank)
    candidate, next_state, accepted = jb.commit_transition(bank, proposal, state, between_steps=True,
                                                          acceptance=lambda _b, _p: True)
    assert accepted and int(candidate.transition_version) == 1
    assert int(bank.transition_version) == 0 and bool(bank.active[1])
    assert not bool(candidate.active[1])
    for kind, entries in next_state.moments.items():
        for name, value in entries.items():
            np.testing.assert_array_equal(np.asarray(value[:2]), np.zeros(value[:2].shape))
            np.testing.assert_array_equal(np.asarray(value[2:]), np.asarray(state.moments[kind][name][2:]))
    assert int(next_state.step) == 7
    optimizer = torch.optim.Adam(torch_bank.parameters())
    for parameter in torch_bank.parameters():
        optimizer.state[parameter] = {"step": torch.tensor(7.0), "exp_avg": torch.ones_like(parameter),
                                      "exp_avg_sq": 2 * torch.ones_like(parameter)}
    assert tb.commit_transition(torch_bank, proposal, optimizer, between_steps=True,
                                acceptance=lambda _b, _p: True)
    assert candidate.portable_snapshot() == torch_bank.portable_snapshot()
    for name, parameter in torch_bank.named_parameters():
        np.testing.assert_array_equal(optimizer.state[parameter]["exp_avg"].numpy(),
                                      np.asarray(next_state.moments["first"][name]))
    x = jnp.linspace(-0.5, 0.5, 9)
    np.testing.assert_allclose(np.asarray(candidate.forward(x)), np.asarray(bank.forward(x)), atol=3e-16)
    np.testing.assert_allclose(torch_bank(torch.tensor(np.asarray(x))).detach().numpy(),
                               np.asarray(candidate.forward(x)), atol=3e-16)
    with pytest.raises(ValueError, match="stale"):
        jb.commit_transition(candidate, proposal, next_state, between_steps=True,
                              acceptance=lambda _b, _p: True)


def test_jit_parameter_gradient_retraction_and_moment_mode() -> None:
    _, bank = pair_banks()
    bank, _, _ = jb.commit_transition(bank, bank.propose_pair(0, 1), moments(bank), between_steps=True,
                                      acceptance=lambda _b, _p: True)
    x = jnp.asarray([-0.3, 0.2])
    compiled = jax.jit(lambda state, z: state.forward(z))
    np.testing.assert_array_equal(np.asarray(compiled(bank, x)), np.asarray(bank.forward(x)))
    def objective(parameters):
        return jnp.sum(bank.with_parameters(parameters).forward(x) ** 2)
    gradients = jax.jit(jax.grad(objective))(bank.parameters)
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in gradients.values())
    assert float(jnp.abs(gradients["odd_moments"][0])) > 0
    boundary = replace(bank, rho=bank.rho.at[0].set(-0.5)).retract()
    assert float(boundary.rho[0]) == 0
    assert float(jax.grad(lambda r: jb.retract_spread(r))(jnp.asarray(0.0))) == 1
    moment_bank = replace(boundary, modes=boundary.modes.at[0].set(2),
                          moment_coefficients=boundary.moment_coefficients.at[0, :2].set(jnp.array([0.7, 0.2])))
    assert bool(jnp.all(jnp.isfinite(compiled(moment_bank, x))))


def test_checkpoint_roundtrip_configuration_binding_and_legacy_migration() -> None:
    torch_bank, bank = pair_banks()
    restored = jb.bank_from_checkpoint(bank.checkpoint())
    assert restored.portable_snapshot() == bank.portable_snapshot()
    bad = deepcopy(bank.checkpoint())
    bad["base"] = "tanh"
    with pytest.raises(ValueError, match="configuration"):
        jb.bank_from_checkpoint(bad)
    bad = deepcopy(bank.checkpoint())
    bad["state"]["modes"]["values"][0] = 8
    with pytest.raises(ValueError, match="representation"):
        jb.bank_from_checkpoint(bad)
    bad = deepcopy(bank.checkpoint())
    bad["state"]["active"]["dtype"] = "float64"
    with pytest.raises(ValueError, match="boolean"):
        jb.bank_from_checkpoint(bad)
    changed_activation = replace(bank, base="tanh")
    with pytest.raises(ValueError, match="stale"):
        jb.commit_transition(changed_activation, bank.propose_pair(0, 1), moments(bank),
                              between_steps=True, acceptance=lambda _b, _p: True)
    packs = [RefinedPack(0, 0.1, 0.7), RefinedPack(1, 0.3, 0.2)]
    legacy = init_pack_bank(packs, max_packs=4, max_order=3, base="sigmoid")
    migrated = jb.migrate_legacy(legacy)
    x = jnp.array([0.1, 0.2])
    np.testing.assert_allclose(np.asarray(migrated.forward(x)), np.asarray(legacy_forward(legacy, x)), atol=1e-16)
    torch_legacy = AdaptivePackBank(packs, max_packs=4, max_order=3, base="sigmoid", dtype=torch.float64)
    torch_bank.load_state_dict(torch_legacy.state_dict())
    assert not bool(torch_bank.modes.any())
    np.testing.assert_allclose(torch_bank(torch.tensor(np.asarray(x))).detach().numpy(),
                               np.asarray(migrated.forward(x)), atol=1e-16)


def test_cluster_row_transition_and_error_gated_derivative() -> None:
    packs = [RefinedPack(0, v, w) for v, w in [(-0.01, 0.7), (0.0, -0.2), (0.02, 0.5)]]
    bank = jb.init_confluent_bank(packs, max_packs=4, max_order=4)
    torch_bank = tb.ConfluentPackBank(packs, max_packs=4, max_order=4, dtype=torch.float64)
    proposal = bank.propose_cluster((0, 1, 2), order=3, error_budget=1e-7)
    assert proposal == torch_bank.propose_cluster((0, 1, 2), order=3, error_budget=1e-7)
    assert TransitionProposal.from_json(proposal.to_json()) == proposal
    candidate, _, accepted = jb.commit_transition(bank, proposal, moments(bank), between_steps=True,
                                                  acceptance=lambda _b, _p: True)
    assert accepted and int(candidate.modes[0]) == 2
    assert candidate.moment_coefficients.shape == (4, 5)
    assert not bool(candidate.active[1]) and not bool(candidate.active[2])
    np.testing.assert_allclose(np.asarray(candidate.forward(jnp.array([-0.2, 0.4]))),
                               np.asarray(bank.forward(jnp.array([-0.2, 0.4]))), atol=2e-9)
    _, pair = pair_banks()
    pair, state, _ = jb.commit_transition(pair, pair.propose_pair(0, 1), moments(pair), between_steps=True,
                                          acceptance=lambda _b, _p: True)
    derivative = pair.propose_derivative(0, error_budget=0)
    same, same_state, accepted = jb.commit_transition(pair, derivative, state, between_steps=True,
                                                     acceptance=lambda _b, p: p.error_budget > 0)
    assert not accepted and same is pair and same_state is state
    assert float(pair.rho[0]) > 0


def test_rollback_and_explicit_optimizer_adapter() -> None:
    torch_bank, bank = pair_banks()
    state = moments(bank)
    proposal = bank.propose_pair(0, 1)
    original, original_state, accepted = jb.commit_transition(bank, proposal, state, between_steps=True,
                                                             acceptance=lambda _b, _p: True,
                                                             postcondition=lambda _b: False)
    assert not accepted and original is bank and original_state is state
    with pytest.raises(TypeError, match="explicit"):
        jb.commit_transition(bank, proposal, {"custom": 3}, between_steps=True,
                              acceptance=lambda _b, _p: True)
    calls = []
    def adapter(custom, candidate, slots):
        calls.append((candidate, slots))
        return {"custom": custom["custom"] + 1}
    _, custom, accepted = jb.commit_transition(bank, proposal, {"custom": 3}, between_steps=True,
                                               acceptance=lambda _b, _p: True, state_adapter=adapter)
    assert accepted and custom == {"custom": 4} and calls[0][1] == (0, 1)
    with pytest.raises(RuntimeError, match="between"):
        jb.commit_transition(bank, proposal, state, between_steps=False, acceptance=lambda _b, _p: True)
    optimizer = torch.optim.Adam(torch_bank.parameters())
    snapshot = torch_bank.portable_snapshot()
    assert not tb.commit_transition(torch_bank, proposal, optimizer, between_steps=True,
                                    acceptance=lambda _b, _p: True, postcondition=lambda _b: False)
    assert torch_bank.portable_snapshot() == snapshot
    assert len(optimizer.state) == 0


@pytest.mark.parametrize("optimizer_name", ["adam", "sgd"])
def test_exact_permutation_moves_ids_parameters_and_moments(optimizer_name: str) -> None:
    torch_bank, bank = pair_banks()
    permutation = (2, 0, 3, 1)
    params_before = {name: id(p) for name, p in torch_bank.named_parameters()}
    source = bank.portable_snapshot()
    optimizer = (torch.optim.Adam(torch_bank.parameters()) if optimizer_name == "adam"
                 else torch.optim.SGD(torch_bank.parameters(), lr=0.01, momentum=0.9))
    first_key = "exp_avg" if optimizer_name == "adam" else "momentum_buffer"
    state = moments(bank)
    unique = {name: jnp.arange(p.size, dtype=p.dtype).reshape(p.shape) + 1
              for name, p in bank.parameters.items()}
    state = replace(state, moments={"first": unique, "second": {name: value ** 2 for name, value in unique.items()}})
    for name, parameter in torch_bank.named_parameters():
        optimizer.state[parameter] = {first_key: torch.tensor(np.asarray(unique[name]))}
        if optimizer_name == "adam":
            optimizer.state[parameter]["step"] = torch.tensor(7.0)
            optimizer.state[parameter]["exp_avg_sq"] = torch.tensor(np.asarray(unique[name] ** 2))
    candidate, next_state = jb.permute_bank(bank, permutation, state, between_steps=True)
    tb.permute_bank(torch_bank, permutation, optimizer, between_steps=True)
    assert candidate.portable_snapshot() == torch_bank.portable_snapshot()
    assert params_before == {name: id(p) for name, p in torch_bank.named_parameters()}
    assert candidate.slot_ids.tolist() == [source["slot_ids"]["values"][i] for i in permutation]
    for name, parameter in torch_bank.named_parameters():
        np.testing.assert_array_equal(optimizer.state[parameter][first_key].numpy(),
                                      np.asarray(next_state.moments["first"][name]))
        if optimizer_name == "adam":
            np.testing.assert_array_equal(optimizer.state[parameter]["exp_avg_sq"].numpy(),
                                          np.asarray(next_state.moments["second"][name]))
            assert float(optimizer.state[parameter]["step"]) == 7
    x = jnp.array([-0.2, 0.1])
    np.testing.assert_allclose(np.asarray(candidate.forward(x)), np.asarray(bank.forward(x)), atol=3e-16)
    with pytest.raises(ValueError, match="every"):
        jb.permute_bank(bank, (0, 0, 1, 2), state, between_steps=True)


def test_tanh_sign_relabel_all_orders_with_moment_parity() -> None:
    packs = [RefinedPack(n, 0.1 * n, 0.3 + n * 0.1, 0.7) for n in range(4)]
    bank = jb.init_confluent_bank(packs, max_packs=4, max_order=3, base="tanh")
    torch_bank = tb.ConfluentPackBank(packs, max_packs=4, max_order=3, base="tanh", dtype=torch.float64)
    state = moments(bank)
    optimizer = torch.optim.Adam(torch_bank.parameters())
    for parameter in torch_bank.parameters():
        optimizer.state[parameter] = {"step": torch.tensor(7.0), "exp_avg": torch.ones_like(parameter),
                                      "exp_avg_sq": 2 * torch.ones_like(parameter)}
    slots = (0, 1, 2, 3)
    candidate, next_state = jb.relabel_tanh_sign(bank, slots, state, between_steps=True)
    tb.relabel_tanh_sign(torch_bank, slots, optimizer, between_steps=True)
    assert candidate.portable_snapshot() == torch_bank.portable_snapshot()
    for name, parameter in torch_bank.named_parameters():
        expected_sign = -1 if name in ("weights", "scales") else 1
        np.testing.assert_array_equal(optimizer.state[parameter]["exp_avg"].numpy(),
                                      np.full(parameter.shape, expected_sign))
        np.testing.assert_array_equal(np.asarray(next_state.moments["first"][name]),
                                      np.full(parameter.shape, expected_sign))
        np.testing.assert_array_equal(np.asarray(next_state.moments["second"][name]),
                                      np.full(parameter.shape, 2))
    x = jnp.array([-0.3, 0.2])
    np.testing.assert_allclose(np.asarray(candidate.forward(x)), np.asarray(bank.forward(x)), atol=2e-14)
    invalid = replace(bank, modes=bank.modes.at[0].set(1))
    with pytest.raises(ValueError, match="ordinary"):
        jb.relabel_tanh_sign(invalid, (0,), state, between_steps=True)
    unknown = replace(state, moments={"unknown_parity": state.moments["first"]})
    with pytest.raises(TypeError, match="parity"):
        jb.relabel_tanh_sign(bank, (0,), unknown, between_steps=True)
    assert bool(jnp.all(bank.scales > 0))


def test_coupled_optimizer_requires_adapter_and_rolls_back() -> None:
    bank, _ = pair_banks()
    optimizer = torch.optim.LBFGS(bank.parameters())
    source = bank.portable_snapshot()
    with pytest.raises(TypeError, match="ExactTransportAdapter"):
        tb.permute_bank(bank, (1, 0, 2, 3), optimizer, between_steps=True)
    assert bank.portable_snapshot() == source


def test_one_registered_tower_per_slot_and_correct_confluent_norms(monkeypatch) -> None:
    torch_bank, bank = pair_banks()
    torch_bank.modes[0] = 1
    torch_bank.modes[1] = 2
    torch_bank.moment_coefficients.data[1, 1] = 0.7
    bank = replace(bank, modes=bank.modes.at[:2].set(jnp.array([1, 2])),
                    moment_coefficients=bank.moment_coefficients.at[1, 1].set(0.7))
    spec = torch_bank.act_spec
    calls = []
    def tower(z, n):
        calls.append(n)
        return spec.tower(z, n)
    torch_bank.act_spec = replace(spec, tower=tower)
    x = torch.tensor([-0.3, 0.2], dtype=torch.float64)
    terms = torch_bank.slot_terms(x)
    assert calls == [11] * torch_bank.max_packs
    np.testing.assert_allclose(np.asarray(bank.slot_terms(jnp.asarray(x.numpy()))), terms.detach().numpy(), atol=3e-16)
    expected = [float(terms[i].detach().square().sum().sqrt()) for i in range(3)]
    assert torch_bank.pack_term_norms(x) == expected
    with pytest.raises(RuntimeError, match="RefinedPack"):
        torch_bank.active_packs()
    with pytest.raises(RuntimeError, match="transactional"):
        torch_bank.sync_from([])
    jspec = jb.get_activation(bank.base)
    jcalls = []
    def jtower(z, n):
        jcalls.append(n)
        return jspec.tower(z, n)
    monkeypatch.setattr(jb, "get_activation", lambda _name: replace(jspec, tower=jtower))
    bank.forward(jnp.asarray(x.numpy()))
    assert jcalls == [11] * bank.max_packs


def test_zero_output_birth_reuses_slot_and_resets_all_moments() -> None:
    torch_bank, bank = pair_banks()
    # Simulate an inactive slot retaining an obsolete collision representation.
    torch_bank.modes[3] = 2
    torch_bank.moment_coefficients.data[3] = 9
    torch_bank.rho.data[3] = 0.2
    bank = replace(bank, modes=bank.modes.at[3].set(2), rho=bank.rho.at[3].set(0.2),
                    moment_coefficients=bank.moment_coefficients.at[3].set(9))
    pack = RefinedPack(2, 0.4, 0, 1.3)
    proposal = bank.propose_birth(3, pack)
    assert proposal == torch_bank.propose_birth(3, pack)
    state = moments(bank)
    candidate, next_state, accepted = jb.commit_transition(bank, proposal, state, between_steps=True,
                                                          acceptance=lambda _b, _p: True)
    optimizer = torch.optim.Adam(torch_bank.parameters())
    for parameter in torch_bank.parameters():
        optimizer.state[parameter] = {"step": torch.tensor(7.0), "exp_avg": torch.ones_like(parameter),
                                      "exp_avg_sq": 2 * torch.ones_like(parameter)}
    assert tb.commit_transition(torch_bank, proposal, optimizer, between_steps=True,
                                acceptance=lambda _b, _p: True)
    assert accepted and candidate.portable_snapshot() == torch_bank.portable_snapshot()
    assert int(candidate.slot_ids[3]) == int(bank.slot_ids[3])
    assert bool(candidate.active[3]) and int(candidate.modes[3]) == 0
    assert bool(jnp.all(candidate.moment_coefficients[3] == 0))
    for entries in next_state.moments.values():
        assert all(bool(jnp.all(value[3] == 0)) for value in entries.values())
    x = jnp.array([-0.2, 0.3])
    np.testing.assert_array_equal(np.asarray(candidate.forward(x)), np.asarray(bank.forward(x)))
    with pytest.raises(ValueError, match="inactive"):
        candidate.propose_birth(3, pack)
    with pytest.raises(ValueError, match="zero output"):
        bank.propose_birth(3, RefinedPack(1, 0.2, 1))
