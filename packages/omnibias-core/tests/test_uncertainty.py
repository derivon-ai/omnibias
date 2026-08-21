# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 04-02: conformal slabs and guarantee-kind separation."""

from __future__ import annotations

import math

import pytest
from omnibias.core.uncertainty import (
    CalibrationReport,
    CombinedStatement,
    GuaranteeKind,
    UncertaintyInterval,
    adaptive_vs_fixed,
    combine_enclosure_with_conformal,
    conformal_index,
    honesty_payload,
    refuse_conformal_seal,
    resample_coverage,
    shift_diagnostic,
    softplus_width,
    softplus_width_prime,
    split_conformal,
    theoretical_coverage,
    worked_example,
)


def test_g1_worked_quantile() -> None:
    ex = worked_example()
    assert ex["q"] == 0.88
    assert conformal_index(19, 0.1) == 18
    naive = int(math.ceil(20 * 0.9))
    assert naive == 18


def test_g1_off_by_one_differs() -> None:
    scores = [float(i) for i in range(1, 21)]
    q = split_conformal(scores, alpha=0.1)
    assert conformal_index(20, 0.1) == 19
    naive_k = int(math.ceil(20 * 0.9))
    assert naive_k == 18
    assert q == scores[18]
    assert q != scores[17]


def test_g1_theory_and_resample() -> None:
    for alpha in (0.01, 0.05, 0.1, 0.2):
        for n in (19, 99, 999):
            assert theoretical_coverage(n, alpha) + 1e-15 >= 1.0 - alpha
    cover = resample_coverage(19, 0.1, trials=400, seed=0)
    assert cover >= 0.84


def test_g2_add_raises() -> None:
    sound = UncertaintyInterval(0.0, 1.0, GuaranteeKind.SOUND_ENCLOSURE)
    conf = UncertaintyInterval(0.0, 1.0, GuaranteeKind.CONFORMAL, level=0.9)
    with pytest.raises(TypeError, match="combine"):
        _ = sound + conf


def test_g3_adaptive() -> None:
    report = adaptive_vs_fixed()
    assert report["g3_earned"] is True
    assert report["high_noise_fixed_cover"] < 0.85


def test_g4_shift() -> None:
    report = shift_diagnostic()
    assert report.shift_detected is True
    assert report.exchangeability_ok is False
    assert report.average_width > 0.0


def test_g5_combine() -> None:
    box = UncertaintyInterval(2.31, 2.47, GuaranteeKind.SOUND_ENCLOSURE)
    stmt = combine_enclosure_with_conformal(box, 0.88, alpha=0.1)
    assert isinstance(stmt, CombinedStatement)
    assert stmt.combined_lo == pytest.approx(2.31 - 0.88)
    assert stmt.enclosure.kind is GuaranteeKind.SOUND_ENCLOSURE
    conf = UncertaintyInterval(-0.88, 0.88, GuaranteeKind.CONFORMAL, level=0.9)
    with pytest.raises(TypeError):
        combine_enclosure_with_conformal(conf, 0.88, alpha=0.1)
    with pytest.raises(ValueError, match="cannot be sealed"):
        refuse_conformal_seal(conf)


def test_g6_average_width_required() -> None:
    report = CalibrationReport(
        marginal_coverage=0.9,
        conditional_coverage={"all": 0.9},
        average_width=1.76,
        width_coverage_curve=((0.1, 0.9),),
    )
    assert report.average_width == 1.76


def test_softplus_prime() -> None:
    assert abs(softplus_width_prime(0.0) - 0.5) < 1e-12
    z = 0.3
    h = 1e-6
    num = (softplus_width(z + h) - softplus_width(z - h)) / (2.0 * h)
    assert abs(num - softplus_width_prime(z)) < 1e-6


def test_honesty() -> None:
    payload = honesty_payload()
    assert payload["registers_blended"] is False
    assert payload["conformal_sealed"] is False
    assert payload["theorem_prover_verified"] is False
