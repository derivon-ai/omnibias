# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact condition-language families owned by omnibias-symbolic."""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.proof import GrammarSpec, apply_constructor, run_discovery
from omnibias.symbolic.conditions import (
    ConservationConditionFamily,
    FractionalOrderFamily,
    JetConditionFamily,
    PdeConditionFamily,
    PiecewiseHybridFamily,
    hypothesis_from_names,
    jet_growth_family,
    square_jet_identity_check,
)
from omnibias.symbolic.lift import planted_heat_rational, snap_sparse_equation, sparse_from_coeffs
from omnibias.symbolic.propose import (
    hypothesis_from_sparse,
    propose_from_residual,
    propose_jet_condition,
)


def test_jet_condition_finds_tanh_riccati() -> None:
    family = JetConditionFamily()
    result = run_discovery(family.statement, family, "score_guided", budget=32)
    assert result.status == "PROVED"
    assert result.check is not None
    assert result.check.payload["honesty"]["discovered_by_omnibias"] is True
    assert result.check.payload["honesty"]["no_condition_exists_claim"] is False
    assert result.equation is not None
    assert "yp" in result.equation.pretty.replace(" ", "")


def test_jet_incomplete_grammar_miss_is_search_incomplete() -> None:
    family = JetConditionFamily()
    family.grammar = GrammarSpec(
        sorts=("jet_monomial",),
        tokens_by_sort={"jet_monomial": ("one",)},
        complete=False,
    )
    empty = hypothesis_from_names("jet_monomial", ("one",))
    checked = family.check(empty)
    assert checked is not None
    # {1} alone is not a unique tanh identity.
    if not checked.ok:
        family_miss = JetConditionFamily()
        family_miss.grammar = GrammarSpec(
            sorts=("jet_monomial",),
            tokens_by_sort={"jet_monomial": ("one", "y")},
            complete=False,
        )
        assert family_miss.complete is False


def test_pde_condition_heat_and_perturbed_snap() -> None:
    family = PdeConditionFamily()
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "PROVED"
    design, target, names = planted_heat_rational()
    bad = snap_sparse_equation(
        sparse_from_coeffs((0.0, 0.0, 0.13), names),
        design,
        target,
        denom_bound=16,
    )
    assert bad is None


def test_conservation_continuity_hits() -> None:
    family = ConservationConditionFamily()
    result = run_discovery(family.statement, family, "score_guided", budget=4)
    assert result.status == "PROVED"
    assert result.check is not None
    assert result.check.payload["honesty"]["no_condition_exists_claim"] is False
    assert result.equation is not None


def test_fractional_integer_order_one_hits() -> None:
    family = FractionalOrderFamily()
    result = run_discovery(family.statement, family, "score_guided", budget=4)
    assert result.status == "PROVED"
    assert result.candidate is not None
    assert result.candidate.coefficients == ("1",)  # type: ignore[union-attr]


def test_piecewise_split_hits_unsigned_misses() -> None:
    family = PiecewiseHybridFamily()
    unsigned = family.check(family.origin())
    assert unsigned is not None
    assert unsigned.ok is False
    result = run_discovery(family.statement, family, "score_guided", budget=4)
    assert result.status == "PROVED"
    assert result.check is not None
    assert result.check.payload["split"] is True


def test_grammar_growth_needs_compose_jets() -> None:
    seed = hypothesis_from_names("jet_monomial", ("y", "yp", "ypp"))
    before = square_jet_identity_check(seed)
    assert before is not None
    assert before.ok is False
    family = jet_growth_family(complete=False)
    grown = apply_constructor(
        seed,
        "compose_jets",
        family.grammar,
        left="yp",
        right="yp",
    )
    assert grown is not None
    after = square_jet_identity_check(grown)
    assert after is not None
    assert after.ok is True
    assert after.payload["annihilator"] is not None
    assert "yp*yp" in after.payload["annihilator"]
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "PROVED"
    depth_two = apply_constructor(grown, "compose_jets", family.grammar, left="y", right="y")
    assert depth_two is None


def test_incomplete_growth_miss_is_search_incomplete() -> None:
    family = jet_growth_family(complete=False)
    family.grammar = GrammarSpec(
        sorts=("jet_monomial",),
        tokens_by_sort={"jet_monomial": ("y", "yp", "ypp")},
        constructors=(),
        complete=False,
    )
    result = run_discovery(family.statement, family, "score_guided", budget=4)
    assert result.status == "BLOCKED"
    assert result.search_incomplete is True


def test_propose_jet_snaps_and_rejects() -> None:
    design, target, names = planted_heat_rational(diffusivity=Fraction(1, 8))
    good = propose_jet_condition(
        sparse_from_coeffs((0.0, 0.0, 0.125), names),
        design,
        target,
        sort="pde_operator",
        kind="pde_span",
    )
    assert good["mode"] == "exact_search"
    assert good["honesty"]["discovered_by_omnibias"] is True
    assert good["statement"] is not None
    hyp = hypothesis_from_sparse(sparse_from_coeffs((0.0, 0.0, 0.125), names), sort="pde_operator")
    assert "u_xx" in hyp.token_names()
    bad = propose_jet_condition(
        sparse_from_coeffs((0.0, 0.0, 0.13), names),
        design,
        target,
        sort="pde_operator",
        kind="pde_span",
    )
    assert bad["mode"] == "empirical"
    assert bad["check"] is None
    assert bad["honesty"]["discovered_by_omnibias"] is False


def test_propose_from_residual_empirical_without_design() -> None:
    payload = propose_from_residual((0.1, 0.2, 0.3))
    assert payload["mode"] == "empirical"
    assert payload["check"] is None
    assert payload["honesty"]["no_condition_exists_claim"] is False
