# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
from __future__ import annotations

from omnibias.dynamics.coalescing_boundary import (
    attack_coalescing_boundary,
    chi_blowup_coordinates,
    verify_coalescing_boundary_report,
)


def test_chi_blowup_coordinates() -> None:
    chi, kappa = chi_blowup_coordinates(2.0, 1.0, 3.0)
    assert chi == 6.0 and kappa == 3.0


def test_coalescing_boundary_campaign_stays_blocked() -> None:
    report = attack_coalescing_boundary()
    assert report.status == "blocked"
    assert verify_coalescing_boundary_report(report)
