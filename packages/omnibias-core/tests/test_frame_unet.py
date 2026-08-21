# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-04: Frame-UNet skip split and denoising skill."""

from __future__ import annotations

from omnibias.core.frame_unet import (
    DISCLAIMER,
    denoise_skill,
    frame_unet_forward,
    honesty_payload,
    worked_example,
)


def test_g1_skip_split() -> None:
    ex = worked_example()
    assert ex["band_err"] < 1e-12
    assert ex["collapse_err"] < 1e-12
    assert ex["skip_gap"] > 1e-3


def test_g2_denoise_skill() -> None:
    report = denoise_skill()
    assert report["below_zero"] is True
    assert report["not_worse_than_scan"] is True


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["sigma_prime_admissible"] is False
    assert payload["stretch_claim"] is False
    assert "not a collapse head" in DISCLAIMER


def test_skips_are_labelled() -> None:
    _y, skips = frame_unet_forward(0.0)
    assert skips["kinds"] == ("band", "collapse")
    bands = skips["band"]
    collapses = skips["collapse"]
    assert isinstance(bands, list) and isinstance(collapses, list)
    assert len(bands) == len(collapses)
    assert any(abs(b - c) > 1e-3 for b, c in zip(bands, collapses, strict=True))
