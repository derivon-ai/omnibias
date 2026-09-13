# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Exact W-identities and ledger bookkeeping; not a C2 or G1 certificate."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.chart_cells import (
    MALETTO_QUARTIC_EXAMPLE,
    g1_from_cells,
    rematch_shrinking_root,
    replay_maletto_type,
)
from omnibias.dynamics.hilbert16_identities import (
    identity_verdict,
    replay_hilbert16_identities,
    residual_outgoing_w_ratio,
    residual_w_height,
)


def test_w_identities_are_exact() -> None:
    eps, w = Fraction(1, 4), Fraction(3, 2)
    assert residual_w_height(eps, w, eps * w) == 0
    assert residual_outgoing_w_ratio(eps, Fraction(4), Fraction(2), eps * 4, eps * 2) == 0


def test_identity_verdict_and_discovery_keep_parent_open() -> None:
    assert identity_verdict("double_root").status == "PROVED"
    result = replay_hilbert16_identities()
    assert result.status == "PROVED"
    assert result.statement.parent_status == "open"
    assert result.check is not None
    assert result.check.payload["honesty"]["full_hilbert16_solved"] is False
    assert result.check.payload["honesty"]["g1_passed"] is False


def test_maletto_example_is_a_combinatorial_type() -> None:
    report = replay_maletto_type(
        MALETTO_QUARTIC_EXAMPLE["counts"],  # type: ignore[arg-type]
        MALETTO_QUARTIC_EXAMPLE["words"],  # type: ignore[arg-type]
        MALETTO_QUARTIC_EXAMPLE["trees"],  # type: ignore[arg-type]
        degree=4,
    )
    assert report["combinatorial_type_ok"] is True
    assert report["algebraic_smoothness"] == "BLOCKED"
    assert report["honesty"]["g1_passed"] is False


def test_sr2_and_ledger_do_not_pass_g1() -> None:
    rematch = rematch_shrinking_root(Fraction(1, 20), Fraction(-2))
    assert rematch["incoming_first_hit_retained"] is True
    assert rematch["first_root_Lmin"] is False
    assert rematch["height_nonnegative_lambda1"] is False
    assert rematch["sr2_rematch"] is False
    assert g1_from_cells() is False
