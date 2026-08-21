# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared-observation binders, logit gate, and STLSQ↔snap bilevel loop."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.core.proof import bind_sorts, select_class
from omnibias.symbolic.classloop import discover_observation, load_discovery_stack, run_bilevel_class_loop
from omnibias.symbolic.conditions import (
    observation_abs,
    observation_heat,
    observation_heat_perturbed,
    observation_square,
    observation_tanh,
)
from omnibias.symbolic.gate import LogitGate
from omnibias.symbolic.ingest import pack_jets, pack_residual, pack_sequence


def test_tanh_observation_selects_jet_monomial() -> None:
    obs = observation_tanh()
    bound = bind_sorts(obs)
    assert "jet_monomial" in bound
    assert "sos_template" not in bound
    assert "forbidden_minor" not in bound
    selection = select_class(obs, sorts=("jet_monomial", "pde_operator"))
    assert selection.best is not None
    assert selection.best.sort == "jet_monomial"
    assert selection.best.tier == "exact"
    assert selection.honesty["no_condition_exists_claim"] is False
    assert selection.grammar_complete is False


def test_heat_design_stlsq_snaps() -> None:
    obs = observation_heat()
    payload = run_bilevel_class_loop(obs, sorts=("pde_operator", "jet_monomial"))
    assert payload["mode"] == "exact_search"
    assert payload["inner"] is not None
    assert payload["inner"]["check"] is True
    assert payload["best"] == "pde_operator"
    assert payload["no_condition_exists_claim"] is False
    assert payload["grammar_complete"] is False


def test_perturbed_design_stays_empirical() -> None:
    obs = observation_heat_perturbed()
    payload = run_bilevel_class_loop(obs, sorts=("pde_operator",))
    assert payload["mode"] == "empirical"
    assert payload["inner"] is not None
    assert payload["inner"]["check"] is None
    selection = payload["selection"]
    assert selection.best is None or selection.best.tier != "exact"
    assert selection.search_incomplete is True
    assert selection.honesty["no_condition_exists_claim"] is False


def test_logit_gate_proposes_jet_but_cannot_forge_sos() -> None:
    obs = observation_tanh()
    gate = LogitGate(sorts=("sos_template", "jet_monomial", "pde_operator"))
    gate.train([(obs, "jet_monomial")])
    assert gate.propose(obs)[0] == "jet_monomial"

    class _SosFirst:
        def propose(self, observation: object) -> tuple[str, ...]:
            del observation
            return ("sos_template", "jet_monomial", "pde_operator")

    forced = select_class(
        obs,
        sorts=("sos_template", "jet_monomial"),
        gate=_SosFirst(),
    )
    assert forced.best is not None
    assert forced.best.sort == "jet_monomial"
    assert forced.honesty["discovered_by_omnibias"] is True


def test_tag_free_square_and_abs_bind() -> None:
    square = observation_square()
    assert square.tag == ""
    bound = bind_sorts(square, ("jet_monomial", "fractional_order", "piecewise_hybrid"))
    assert "fractional_order" in bound
    assert "jet_monomial" in bound
    abs_obs = observation_abs()
    assert abs_obs.tag == ""
    piecewise = bind_sorts(abs_obs, ("piecewise_hybrid", "jet_monomial"))
    assert "piecewise_hybrid" in piecewise


def test_pde_requires_second_derivative_name() -> None:
    from omnibias.symbolic.ingest import pack_design

    obs = pack_design(((1, 2),), (0,), ("u", "u_x"))
    assert bind_sorts(obs, ("pde_operator",)) == {}


def test_grow_on_square_jets() -> None:
    obs = observation_square()
    miss = select_class(obs, sorts=("jet_monomial",), grow=False, inner_budget=4)
    assert miss.best is None or miss.best.tier != "exact" or "*" not in str(
        miss.best.check.payload.get("annihilator", "")
    )
    grown = select_class(obs, sorts=("jet_monomial",), grow=True, inner_budget=4)
    assert grown.best is not None
    assert grown.best.tier == "exact"
    assert grown.honesty["no_condition_exists_claim"] is False
    assert grown.grammar_complete is False


def test_residual_sign_enclosure() -> None:
    obs = pack_residual((1, 2, Fraction(1, 2)))
    bound = bind_sorts(obs, ("residual_sign",))
    assert "residual_sign" in bound
    selection = select_class(obs, sorts=("residual_sign",))
    assert selection.best is not None
    assert selection.best.tier == "enclosure"


def test_pack_sequence_binds_without_tag() -> None:
    obs = pack_sequence((0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610))
    assert obs.tag == ""
    try:
        import omnibias.holonomic.conditions  # noqa: F401
    except ImportError:
        return
    bound = bind_sorts(obs, ("ore", "dfinite"))
    assert "ore" in bound
    assert "dfinite" not in bound


def test_field_fit_optional_hidden_eight() -> None:
    pytest.importorskip("jax")
    obs = observation_tanh()
    payload = run_bilevel_class_loop(
        obs,
        sorts=("jet_monomial",),
        proposers=("field",),
        fit_field=True,
        field_hidden=8,
    )
    assert payload["no_condition_exists_claim"] is False
    assert payload["grammar_complete"] is False


def test_discover_observation_tanh() -> None:
    load_discovery_stack()
    payload = discover_observation(
        observation_tanh(),
        sorts=("jet_monomial", "pde_operator"),
        proposers=("neural_jet",),
    )
    assert payload["no_condition_exists_claim"] is False
    assert payload["grammar_complete"] is False
    assert payload["best"] == "jet_monomial"


def test_ingest_jets_without_tag() -> None:
    obs = pack_jets(
        {"y": (1, 4, 9), "yp": (2, 4, 6), "ypp": (2, 2, 2)},
        sample_x=(1, 2, 3),
    )
    assert obs.tag == ""
    assert "fractional_order" in bind_sorts(obs, ("fractional_order", "jet_monomial"))
