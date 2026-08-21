# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-07: exact oscillator ladder (tooling, not a many-body solution)."""

from __future__ import annotations

import math

import pytest
from omnibias.ferminet.hermite import (
    DISCLAIMER,
    apply_ladder,
    gaussian_envelope_energy,
    hermite_ladder_basis,
    honesty_payload,
    ladder_coefficient_errors,
    oscillator_phi,
    qho_ground_energy,
    vmc_comparison_report,
)


def test_g0_qho_textbook_energy() -> None:
    assert qho_ground_energy() == 0.5
    assert abs(gaussian_envelope_energy(1.0) - 0.5) <= 1e-15
    assert honesty_payload()["many_body_solution_claim"] is False
    assert "not a many-body solution" in DISCLAIMER


def test_g1_ladder_coefficients() -> None:
    errs = ladder_coefficient_errors()
    assert errs
    assert max(errs) <= 2e-14


def test_g2_reports_no_improvement_against_competent_baseline() -> None:
    report = vmc_comparison_report(seeds=5)
    assert report["no_improvement"] is True
    assert abs(float(report["ladder_energy"]) - 0.5) <= 1.6e-3
    assert report["ladder_steps"] == 0


def test_normalization_required() -> None:
    with pytest.raises(TypeError):
        hermite_ladder_basis(4)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        hermite_ladder_basis(4, normalization="silent")
    basis = hermite_ladder_basis(4, normalization="oscillator")
    assert basis.eval(0, 0.0) == pytest.approx(oscillator_phi(0, 0.0))
    lowered = apply_ladder(1, 0.5, "lower", normalization="oscillator")
    assert lowered == pytest.approx(math.sqrt(1.0) * oscillator_phi(0, 0.5), rel=0, abs=1e-14)
