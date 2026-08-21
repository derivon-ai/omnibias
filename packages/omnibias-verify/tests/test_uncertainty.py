# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Verify import path for theory 04-02."""

from __future__ import annotations

from omnibias.verify.uncertainty import (
    GuaranteeKind,
    UncertaintyInterval,
    refuse_conformal_seal,
    worked_example,
)


def test_verify_reexport() -> None:
    assert worked_example()["q"] == 0.88
    conf = UncertaintyInterval(-1.0, 1.0, GuaranteeKind.CONFORMAL, level=0.9)
    try:
        refuse_conformal_seal(conf)
    except ValueError as exc:
        assert "cannot be sealed" in str(exc)
    else:
        raise AssertionError("conformal seal must raise")
