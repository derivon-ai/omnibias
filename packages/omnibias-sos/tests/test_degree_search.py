# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""SOS degree-box search on a planted nonnegative polynomial."""

from __future__ import annotations

from omnibias.core.proof import Conjecture, run_discovery
from omnibias.sos.families import SosDegreeFamily
from omnibias.sos.proofmachine import SOS_DEGREE_SEARCH, build_sos_machine


def test_sos_degree_family_hits() -> None:
    family = SosDegreeFamily()
    result = run_discovery(family.statement, family, "score_guided", budget=4)
    assert result.status == "PROVED"
    assert result.check is not None
    assert result.check.payload["honesty"]["unproven_claim"] is False


def test_sos_machine_degree_kind() -> None:
    machine = build_sos_machine()
    verdict = machine.evaluate(Conjecture(name="deg", kind=SOS_DEGREE_SEARCH))
    assert verdict.status == "PROVED"
