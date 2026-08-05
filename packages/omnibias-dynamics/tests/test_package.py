# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Smoke tests for the omnibias-dynamics package scaffold."""

from __future__ import annotations

import omnibias.dynamics as dyn


def test_version() -> None:
    assert dyn.__version__ == "0.1.0a1"


def test_depends_only_on_core() -> None:
    # The dynamics core must build on the rigorous core, never on a backend.
    import omnibias.core.verified  # noqa: F401  (import guard)
