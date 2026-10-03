# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
from __future__ import annotations

from omnibias.dynamics.g1_passage import (
    existing_section_log_w_ratio,
    frozen_exponent_kill_sequence_report,
    frozen_majorant_log_ratio,
)
from omnibias.dynamics.scale_dichotomy import existing_section_explosion


def test_frozen_exponent_diverges_on_kill_sequence() -> None:
    report = frozen_exponent_kill_sequence_report()
    assert report.frozen_exceeds_one
    assert report.route_specific


def test_existing_section_log_w_ratio_depends_on_sigma() -> None:
    low = existing_section_log_w_ratio(0.05, 4, 0.5)
    high = existing_section_log_w_ratio(0.05, 4, 2.0)
    assert low != high


def test_frozen_majorant_is_sensitive_to_gamma() -> None:
    base = frozen_majorant_log_ratio(0.1, 1.0, 10.0, 1.0, 1.0)
    shifted = frozen_majorant_log_ratio(0.1, 1.0, 10.0, 1.0, 0.5)
    assert base != shifted


def test_scale_dichotomy_keeps_new_closing_map_false() -> None:
    payload = existing_section_explosion()
    assert payload["new_closing_map"] is False
    assert payload["closing_map_tautological"] is True
