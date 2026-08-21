# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""SOS-template condition family."""

from __future__ import annotations

from omnibias.core.proof import Observation, bind_sorts, run_discovery
from omnibias.sos.conditions import (
    SosOnsetFamily,
    SosTemplateFamily,
    bind_sos_onset,
    bind_sos_template,
    observation_planted_onset,
    observation_planted_sos,
)
from omnibias.sos.problem import Polynomial


def test_sos_template_planted_hits() -> None:
    family = SosTemplateFamily()
    result = run_discovery(family.statement, family, "score_guided", budget=4)
    assert result.status == "PROVED"
    assert result.check is not None
    assert result.check.payload["honesty"]["no_condition_exists_claim"] is False
    assert result.check.payload["honesty"]["discovered_by_omnibias"] is True


def test_poly_only_observation_binds_only_sos() -> None:
    obs = observation_planted_sos()
    bound = bind_sorts(obs)
    assert set(bound) == {"sos_template"}
    family = bind_sos_template(obs)
    assert family is not None
    result = run_discovery(family.statement, family, "score_guided", budget=4)
    assert result.status == "PROVED"
    assert bind_sos_template(Observation(tag="tanh")) is None
    assert bind_sos_onset(obs) is None


def test_sos_onset_planted_enclosure() -> None:
    obs = observation_planted_onset()
    family = bind_sos_onset(obs)
    assert family is not None
    result = run_discovery(family.statement, family, "score_guided", budget=4)
    assert result.status == "PROVED"
    assert result.check is not None
    assert result.check.payload.get("tier") == "enclosure"
    catalog = SosOnsetFamily()
    replay = run_discovery(catalog.statement, catalog, "score_guided", budget=4)
    assert replay.status in {"PROVED", "BLOCKED"}


def test_sos_template_vanishing_poly_misses() -> None:
    x = Polynomial.variable(0, 2)
    y = Polynomial.variable(1, 2)
    family = SosTemplateFamily(polynomial=x * x + y * y)
    result = run_discovery(family.statement, family, "score_guided", budget=4)
    assert result.status == "BLOCKED"
    assert result.search_incomplete is True
    assert result.check is None or result.check.ok is False
