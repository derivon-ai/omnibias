# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The certified schedule satisfies the permissive tropical operator protocol."""

from omnibias.discrete import AnnealSchedule
from omnibias.struct._core.tropical import as_tropical_schedule


def test_discrete_schedule_drives_tropical_operators() -> None:
    schedule = AnnealSchedule.fast()
    tropical = as_tropical_schedule(schedule)
    assert tropical.betas() == schedule.betas()
