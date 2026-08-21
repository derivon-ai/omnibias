# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-13: Jet-Hopfield memory."""

from __future__ import annotations

import pytest
from omnibias.core.jet_hopfield import (
    DISCLAIMER,
    JetHopfieldConfig,
    contact_split,
    contact_sq,
    honesty_payload,
    jet_hopfield_retrieve,
    nearest_index,
    worked_example,
)


def test_g1_near_j1() -> None:
    ex = worked_example()
    assert ex["value_err"] < 1e-6
    assert abs(ex["d2_j1"] - 1e-4) < 1e-15
    assert abs(ex["d2_j2"] - 1.9801) < 1e-12


def test_g2_contact_split() -> None:
    report = contact_split()
    assert report["split"] is True
    assert report["lam"] == 1.0


def test_g3_honesty() -> None:
    assert honesty_payload()["temperature_collapse_used"] is False
    assert honesty_payload(beta=float("inf"))["temperature_collapse_used"] is True
    assert "not vector" in DISCLAIMER


def test_value_only_still_picks_j1_on_g1() -> None:
    cfg = JetHopfieldConfig(lam=0.0, beta=10.0)
    assert nearest_index((1.0, 0.01), ((1.0, 0.0), (0.0, 1.0)), config=cfg) == 0


def test_domain() -> None:
    with pytest.raises(ValueError, match="lam"):
        contact_sq((1.0, 0.0), (0.0, 1.0), config=JetHopfieldConfig(lam=-1.0))
    with pytest.raises(ValueError, match="jet"):
        jet_hopfield_retrieve((1.0,), ((1.0, 0.0),), config=JetHopfieldConfig())
