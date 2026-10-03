# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
from __future__ import annotations

from omnibias.dynamics.df1a import (
    DF1A_CYCLICITY_BOUND,
    reproduce_df1a_cyclicity,
    verify_df1a_certificate,
)


def test_df1a_replay_respects_published_bound() -> None:
    certificate = reproduce_df1a_cyclicity()
    assert verify_df1a_certificate(certificate)
    assert certificate.cyclicity_bound <= DF1A_CYCLICITY_BOUND
    assert certificate.seal["honesty"]["df1a_independently_replayed"] is True
    assert certificate.seal["honesty"]["full_hilbert16_solved"] is False
