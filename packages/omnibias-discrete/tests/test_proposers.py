# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Optional anneal_descent proposer (not a core get_proposer name)."""

from __future__ import annotations

import pytest
from omnibias.core.proof import IntegerIntervalFamily, get_proposer, run_discovery
from omnibias.discrete.proposers import AnnealDescentProposer


def test_core_get_proposer_still_rejects_anneal_descent() -> None:
    try:
        get_proposer("anneal_descent")
    except ValueError as exc:
        assert "unknown proposer" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_anneal_proposer_hits_toy_via_fallback() -> None:
    pytest.importorskip("torch")
    family = IntegerIntervalFamily(lo=-3, hi=3, target_square=4)
    result = run_discovery(
        family.statement,
        family,
        AnnealDescentProposer(),
        budget=16,
    )
    assert result.status == "PROVED"
    assert result.proposer == "anneal_descent"
    assert result.candidate in (-2, 2)
