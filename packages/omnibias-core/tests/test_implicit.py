# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-free implicit DEQ algebra (theory 08-08)."""

from __future__ import annotations

import pytest
from omnibias.core.implicit import (
    DEQConfig,
    DEQNotContractive,
    DEQSolverUnknown,
    honesty_payload,
    reject_anderson,
    reject_deq_contraction,
    spectral_radius_inf_bound,
)


def test_config_rejects_bad_budget() -> None:
    with pytest.raises(ValueError, match="max_iter"):
        DEQConfig(max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        DEQConfig(tol=0.0)
    with pytest.raises(ValueError, match="solver"):
        DEQConfig(solver="picard")  # type: ignore[arg-type]


def test_contraction_raise() -> None:
    reject_deq_contraction(0.5, require=True)
    with pytest.raises(DEQNotContractive, match="unroll"):
        reject_deq_contraction(1.0, require=True)
    reject_deq_contraction(1.5, require=False)


def test_anderson_is_extra() -> None:
    with pytest.raises(DEQSolverUnknown, match="newton"):
        reject_anderson("anderson")
    reject_anderson("newton")


def test_honesty_sealed() -> None:
    payload = honesty_payload()
    assert payload["navier_stokes_proof_claim"] is False
    assert payload["ccf_stretch_cleared"] is False
    assert payload["unrolled_bptt_claim"] is False
    payload["ccf_stretch_cleared"] = True
    assert honesty_payload()["ccf_stretch_cleared"] is False


def test_inf_bound() -> None:
    assert spectral_radius_inf_bound(0.141) == pytest.approx(0.141)
    with pytest.raises(ValueError):
        spectral_radius_inf_bound(-0.1)
