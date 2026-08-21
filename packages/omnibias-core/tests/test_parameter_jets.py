# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-27: parameter-space mixed jets."""

from __future__ import annotations

import pytest
from omnibias.core.parameter_jets import (
    DISCLAIMER,
    ParameterJetSpec,
    honesty_payload,
    mixed_jet,
    parameter_jet_skill,
    worked_example,
)


def test_g1_fourier_heat() -> None:
    ex = worked_example()
    assert ex["abs_sum"] < 1e-8


def test_g2_skill() -> None:
    report = parameter_jet_skill()
    assert report["g2_earned"] is True
    assert report["beats_fd"] is True


def test_g3_closed_form_requires_trunk() -> None:
    spec = ParameterJetSpec(method="closed_form", mu_in_jet_trunk=False)
    with pytest.raises(ValueError, match="mu_in_jet_trunk"):
        mixed_jet(None, (1.0, 1.0), 1.0, spec=spec)
    ad = ParameterJetSpec(method="autodiff", mu_in_jet_trunk=False)
    assert ad.method != "closed_form"


def test_g4_honesty() -> None:
    payload = honesty_payload()
    assert payload["parampinn_package"] is False
    assert payload["ns_claim"] is False
    assert payload["stretch_claim"] is False
    assert "not a ParamPINN" in DISCLAIMER
