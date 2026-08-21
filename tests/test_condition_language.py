# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-package condition-language catalog and meta-family."""

from __future__ import annotations

from omnibias.core.proof import (
    KindMetaFamily,
    bind_sorts,
    discover,
    list_catalog,
    run_discovery,
    select_class,
)


def test_condition_catalog_kinds_unique() -> None:
    try:
        import omnibias.combinatorics.proofmachine  # noqa: F401
        import omnibias.holonomic.proofmachine  # noqa: F401
        import omnibias.sos.proofmachine  # noqa: F401
        import omnibias.symbolic.proofmachine  # noqa: F401
    except ImportError:
        return
    kinds = [entry.kind for entry in list_catalog()]
    assert len(kinds) == len(set(kinds))
    expected = {
        "condition_jet_monomial",
        "condition_pde_operator",
        "condition_conservation",
        "condition_fractional_order",
        "condition_piecewise_hybrid",
        "condition_kind_meta",
        "condition_grammar_growth",
        "condition_ore",
        "condition_sos_template",
        "condition_forbidden_minor",
        "condition_dfinite",
        "condition_sos_onset",
        "condition_edge_colouring",
        "condition_extremal_template",
        "condition_residual_sign",
    }
    assert expected <= set(kinds)


def test_kind_meta_jet_hits_other_sorts_do_not_false_accept() -> None:
    try:
        import omnibias.combinatorics.minors  # noqa: F401
        import omnibias.holonomic.conditions  # noqa: F401
        import omnibias.sos.conditions  # noqa: F401
        from omnibias.symbolic.conditions import observation_tanh
    except ImportError:
        return
    obs = observation_tanh()
    bound = bind_sorts(obs)
    assert "jet_monomial" in bound
    assert "sos_template" not in bound
    assert "forbidden_minor" not in bound
    meta = KindMetaFamily(
        sorts=("jet_monomial", "sos_template", "forbidden_minor"),
        observation=obs,
        grammar_complete=True,
        budget=32,
    )
    assert meta.complete is False
    result = run_discovery(meta.statement, meta, "score_guided", budget=8, collect=True)
    assert result.status == "PROVED"
    selection = select_class(
        obs,
        sorts=("jet_monomial", "sos_template", "forbidden_minor"),
    )
    assert selection.best is not None
    assert selection.best.sort == "jet_monomial"
    assert selection.honesty["no_condition_exists_claim"] is False
    assert selection.grammar_complete is False


def test_discover_condition_jet() -> None:
    try:
        import omnibias.symbolic.proofmachine  # noqa: F401
    except ImportError:
        return
    result = discover("condition_jet_monomial", budget=32)
    assert result.status == "PROVED"
