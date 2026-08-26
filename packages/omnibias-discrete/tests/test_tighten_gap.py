# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""tighten_gap keeps the best sound Lasserre floor and never claims tightness."""

from __future__ import annotations

import pytest
from omnibias.discrete import (
    TightenedGap,
    brute_force_min,
    decode,
    tighten_gap,
)
from omnibias.discrete.maxsat import max_sat


def test_tighten_gap_sandwiches_and_never_claims_tight() -> None:
    pytest.importorskip("omnibias.sos")
    prob = max_sat([[1, -2], [2, 3], [-1, -3]], weights=[1.5, 2.0, 0.5])
    _, e_min = brute_force_min(prob)
    assignment, _ = decode(prob, n_starts=16)
    tightened = tighten_gap(prob, assignment, levels=(1, 2), bisection_steps=12)
    assert isinstance(tightened, TightenedGap)
    assert tightened.tight is False
    assert tightened.honesty()["p_vs_np_claim"] is False
    assert tightened.honesty()["tight"] is False
    cert = tightened.certificate
    assert cert.is_sound
    assert cert.lower_bound <= e_min + 1e-6
    assert cert.energy >= e_min - 1e-9
    assert tightened.level_used in tightened.levels_tried
    assert tightened.gap == pytest.approx(cert.absolute_gap)


def test_tighten_gap_level_2_not_worse_than_level_1() -> None:
    pytest.importorskip("omnibias.sos")
    prob = max_sat([[1], [-1]])
    assignment, _ = decode(prob)
    one = tighten_gap(prob, assignment, levels=(1,), bisection_steps=12)
    both = tighten_gap(prob, assignment, levels=(1, 2), bisection_steps=12)
    assert both.certificate.lower_bound + 1e-9 >= one.certificate.lower_bound
    assert both.tight is False
