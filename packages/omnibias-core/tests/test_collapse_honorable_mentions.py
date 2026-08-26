# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Honorable mentions: evaluated and refused; no modules shipped.

Cocycle, path, q-specialization, scale/RG, Morse, and policy collapse
do not mint a new surviving object. They are compositions, already
shipped limits, or temperature collapse of a search heuristic.
"""

from __future__ import annotations

import pytest
from omnibias.core.collapse import (
    FOUNDING_SURVIVING,
    CollapseSpec,
    list_collapses,
    register_collapse,
    reset_collapse_registry,
)


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    reset_collapse_registry()
    yield
    reset_collapse_registry()


def _spec(name: str, parameter: str, surviving_object: str) -> CollapseSpec:
    return CollapseSpec(
        name=name,
        parameter=parameter,
        limit="0",
        surviving_object=surviving_object,
        failure="Inconclusive",
        home="omnibias.core.collapse.schema",
        register="verified",
    )


def test_founding_surviving_objects_are_reserved() -> None:
    assert FOUNDING_SURVIVING == {"derivative", "indicator", "point_plus_proof"}
    with pytest.raises(ValueError, match="reuses founding surviving object"):
        register_collapse(_spec("specialization", "q", "derivative"))
    with pytest.raises(ValueError, match="reuses founding surviving object"):
        register_collapse(_spec("policy_as_temperature", "proposer_entropy", "indicator"))


def test_honorable_mentions_are_not_shipped() -> None:
    names = {spec.name for spec in list_collapses()}
    refused = {
        "cocycle",
        "path",
        "specialization",
        "scale",
        "morse",
        "policy",
        "gap",
    }
    assert names.isdisjoint(refused)
    from omnibias.core import collapse as package

    for attr in (
        "cocycle_collapse",
        "path_collapse",
        "specialization_collapse",
        "scale_collapse",
        "morse_collapse",
        "policy_collapse",
        "gap_collapse",
    ):
        assert not hasattr(package, attr)
