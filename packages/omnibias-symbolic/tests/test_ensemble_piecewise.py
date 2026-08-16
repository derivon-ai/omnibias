# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Piecewise confined / deconfined Path B discoverer."""

from __future__ import annotations

import pytest
from omnibias.symbolic.ensemble_piecewise import (
    PiecewiseEnsembleDiscoverer,
    planted_hybrid_wilson_table,
)


def test_piecewise_beats_global_area_law() -> None:
    table = planted_hybrid_wilson_table()
    out = PiecewiseEnsembleDiscoverer().fit(table, threshold=1.0)
    assert out.passed is True
    assert out.skill > 0.0
    assert len(out.automaton.laws) >= 2
    assert out.yang_mills_claim is False
    assert out.continuum_claim is False
