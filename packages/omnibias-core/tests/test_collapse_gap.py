# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Duality / gap collapse was evaluated and rejected as a rebrand.

A primal-dual sandwich ``[L, U]`` that collapses at ``L = U`` is Enclosure
Collapse: ``width -> 0`` of a sound enclosure, yielding a point plus a proof
for ``OPT``. Renaming the parameter does not mint a new surviving object. This
file keeps the refusal, not the module.
"""

from __future__ import annotations

import pytest
from omnibias.core.collapse import (
    CollapseSpec,
    register_collapse,
    reset_collapse_registry,
)


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    reset_collapse_registry()
    yield
    reset_collapse_registry()


def test_gap_as_enclosure_rebrand_is_refused() -> None:
    clone = CollapseSpec(
        name="gap",
        parameter="width",
        limit="0",
        surviving_object="point_plus_proof",
        failure="Inconclusive",
        home="omnibias.core.collapse.schema",
        register="verified",
    )
    with pytest.raises(ValueError, match="rebrand of 'enclosure'"):
        register_collapse(clone)


def test_renamed_gap_parameter_is_not_shipped() -> None:
    from omnibias.core import collapse as package

    assert not hasattr(package, "gap_collapse")
    assert not hasattr(package, "GAP_SPEC")
