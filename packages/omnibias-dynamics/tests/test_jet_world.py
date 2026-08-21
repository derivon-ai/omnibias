# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Dynamics re-export of the 09-25 jet world-model."""

from __future__ import annotations

from omnibias.dynamics import JetWorldConfig, predict_next_jet
from omnibias.dynamics._core.jet_world import worked_example


def test_reexport_g1() -> None:
    cfg = JetWorldConfig()
    assert predict_next_jet(None, None, config=cfg) == 1.0 - cfg.dt * cfg.dt / 2.0
    assert worked_example()["abs_err"] < 1e-12
