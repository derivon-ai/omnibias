# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-06: coupling jet-flow log-det and Newton inverse."""

from __future__ import annotations

from omnibias.core.coupling_flow import (
    DISCLAIMER,
    honesty_payload,
    mixture_skill,
    roundtrip_grid,
    worked_example,
)


def test_g1_log_det() -> None:
    ex = worked_example()
    assert ex["log_det_err"] < 1e-12
    assert ex["inv_err"] < 1e-12


def test_g2_roundtrip() -> None:
    assert roundtrip_grid() < 1e-10


def test_g3_mixture_beats_isotropic() -> None:
    report = mixture_skill()
    assert report["beats_iso"] is True
    assert float(report["skill"]) > 0.0


def test_honesty() -> None:
    payload = honesty_payload()
    assert payload["imagenet_claim"] is False
    assert payload["rewrites_integrate_cnf"] is False
    assert "not integrate_cnf" in DISCLAIMER
