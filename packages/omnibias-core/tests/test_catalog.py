# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Discovery catalog: register, collide, discover, imported-packages-only."""

from __future__ import annotations

from omnibias.core.proof import (
    CatalogEntry,
    IntegerIntervalFamily,
    catalog_entry,
    discover,
    list_catalog,
    register_catalog,
    run_discovery,
)


def _toy_entry(kind: str = "integer_square_toy") -> CatalogEntry:
    return CatalogEntry(
        kind=kind,
        obligation="some integer x in the interval with x^2 equal to the target",
        parent="toy",
        parent_status="already_true",
        package="omnibias.core.proof",
        mode="exact_search",
        complete=True,
    )


def _toy_factory(*, lo: int = -3, hi: int = 3, budget: int = 8) -> object:
    family = IntegerIntervalFamily(lo=lo, hi=hi, target_square=4)
    return run_discovery(family.statement, family, "score_guided", budget=budget)


def test_register_and_discover_toy() -> None:
    register_catalog(_toy_entry(), _toy_factory)
    kinds = {item.kind for item in list_catalog()}
    assert "integer_square_toy" in kinds
    result = discover("integer_square_toy")
    assert result.status == "PROVED"
    assert result.candidate in (-2, 2)


def test_same_entry_is_idempotent() -> None:
    entry = _toy_entry()
    register_catalog(entry, _toy_factory)
    register_catalog(entry, _toy_factory)
    assert catalog_entry("integer_square_toy") == entry


def test_kind_collision_raises() -> None:
    kind = "integer_square_toy_collision"
    register_catalog(_toy_entry(kind))
    try:
        register_catalog(
            CatalogEntry(
                kind=kind,
                obligation="a different obligation",
                parent="toy",
                parent_status="already_true",
                package="omnibias.core.proof",
                mode="exact_search",
                complete=True,
            )
        )
    except ValueError as exc:
        assert "collision" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_discover_without_factory_names_package() -> None:
    kind = "integer_square_toy_nofactory"
    register_catalog(_toy_entry(kind))
    try:
        discover(kind)
    except KeyError as exc:
        assert "omnibias.core.proof" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_unknown_kind_names_owning_package() -> None:
    try:
        discover("not_a_kind")
    except KeyError as exc:
        assert "the owning package" in str(exc)
    else:
        raise AssertionError("expected KeyError")
