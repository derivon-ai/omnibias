# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""AFP / Rybin DGG replay and blind H* search."""

from __future__ import annotations

from pathlib import Path

from omnibias.combinatorics.unsplittable import (
    matches_known_replay,
    search_dgg,
    search_hit_certificate,
    verify_rybin_instance,
)
from omnibias.combinatorics.unsplittable_dags import (
    evaluate_dag,
    planted_separator,
    search_dgg_dags,
)
from omnibias.combinatorics.unsplittable_search import evaluate_hstar


def test_rybin_replay_separates_58_from_60() -> None:
    cert = verify_rybin_instance()
    assert cert["replay_ok"] is True
    assert cert["fractional_cost"] == "58"
    assert cert["min_legal_unsplittable"] == "60"
    assert cert["honesty"]["dgg_congestion_theorem_refuted"] is False
    assert cert["honesty"]["discovered_by_omnibias"] is False
    assert cert["honesty"]["dgg_cost_conjecture_replay"] is True


def test_planted_recovery() -> None:
    planted = (
        ((8, 8, 8), (0, 0, 0), (4, 4, 4)),
        ((8, 8, 8), (1, 0, 0), (4, 4, 4)),
        ((15, 10, 15), (2, 3, 2), (10, 6, 10)),
    )
    hits = search_dgg(family="planted", instances=planted, max_hits=1)
    assert hits
    assert hits[0].fractional_cost < hits[0].min_legal


def test_blind_hstar_box_finds_a_separator() -> None:
    hits = search_dgg(family="hstar_parameter_box", max_hits=1)
    assert hits
    payload = search_hit_certificate(hits[0])
    assert payload["replay_ok"] is True
    assert payload["honesty"]["dgg_congestion_theorem_refuted"] is False
    if matches_known_replay(hits[0]):
        assert payload["rediscovered_known_witness"] is True
        assert payload["discovered_by_omnibias"] is False
    else:
        assert payload["discovered_by_omnibias"] is True


def test_search_module_is_blind() -> None:
    src = Path(__file__).resolve().parents[1] / "src/omnibias/combinatorics/unsplittable_search.py"
    text = src.read_text(encoding="utf-8")
    assert " 58" not in text and "58," not in text and "= 58" not in text
    assert " 60" not in text and "60," not in text and "= 60" not in text
    assert "rybin_instance" not in text


def test_evaluate_hstar_rejects_zero_cost() -> None:
    from fractions import Fraction

    assert evaluate_hstar((Fraction(8),) * 3, (Fraction(0),) * 3, (Fraction(4),) * 3) is None


def test_dag_le6_ci_miss_is_blocked() -> None:
    hits = search_dgg_dags(family="three_terminal_dag_le6")
    assert hits == []


def test_dag_planted_recovery() -> None:
    dag, demands, costs, split = planted_separator()
    from fractions import Fraction

    found = evaluate_dag(
        dag,
        tuple(Fraction(v) for v in demands),
        tuple(Fraction(v) for v in costs),
        tuple(Fraction(v) for v in split),
    )
    assert found is not None
    assert found["fractional_cost"] < found["min_legal"]
    assert len(dag.vertices) <= 6
    hits = search_dgg_dags(family="planted")
    assert hits
    assert hits[0].topology == dag.name
    assert hits[0].fractional_cost < hits[0].min_legal


def test_optional_proposers_on_hstar() -> None:
    for proposer in ("coordinate_newton", "onehot_anneal"):
        hits = search_dgg(
            family="hstar_parameter_box", proposer=proposer, budget=64, max_hits=1
        )
        if hits:
            assert hits[0].fractional_cost < hits[0].min_legal
            return
    hits = search_dgg(family="hstar_parameter_box", max_hits=1)
    assert hits
