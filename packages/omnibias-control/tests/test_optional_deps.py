# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Optional-dependency degradation for ``omnibias-control``.

``omnibias-control`` hard-depends only on ``omnibias-core`` + numpy. The rigorous
recoverable-set certificate reuses the optional ``omnibias-verify`` extra (imported
lazily inside :func:`omnibias.control.certify_recoverable`). The contract is: when
``omnibias-verify`` is absent the certificate helpers **degrade gracefully to
``None``** -- they never raise at import time and never crash the caller. These
tests simulate the missing extra with a blocked ``sys.modules`` entry so the
degradation path is exercised even in an environment where the extra is installed.
"""

from __future__ import annotations

import sys

import pytest
from omnibias.control import (
    certify_disc_recoverable,
    certify_recoverable,
    disc_obstacle_margin,
)

GAINS = (2.0, 2.0)
AMAX = 2.5
R = 1.0
CENTER = (0.0, 0.0)
SAFE_RANGES = [(-1.5, 1.5), (-2.5, -1.5)]


def _block_import(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make ``import <name>`` / ``from <name> import ...`` raise ImportError.

    A ``None`` entry in ``sys.modules`` is the canonical "this import is blocked"
    marker honoured by CPython's import machinery.
    """
    monkeypatch.setitem(sys.modules, name, None)


def test_certify_disc_recoverable_degrades_to_none_without_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_import(monkeypatch, "omnibias.verify")
    cert = certify_disc_recoverable(CENTER, R, GAINS, AMAX, SAFE_RANGES, 0.5, tol=1e-2)
    assert cert is None


def test_certify_recoverable_degrades_to_none_without_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnibias.core.verified import Interval

    phi, grad = disc_obstacle_margin(CENTER, R, GAINS, AMAX)
    box = [
        Interval(-1.5, 1.5),
        Interval(-2.5, -1.5),
        Interval(-0.5, 0.5),
        Interval(-0.5, 0.5),
    ]
    _block_import(monkeypatch, "omnibias.verify")
    assert certify_recoverable(phi, box, grad=grad, tol=1e-2) is None


def test_certificate_is_produced_when_verify_is_installed() -> None:
    """Positive path: with the extra present the helper certifies (non-vacuous guard)."""
    pytest.importorskip("omnibias.verify")
    cert = certify_disc_recoverable(CENTER, R, GAINS, AMAX, SAFE_RANGES, 0.5, tol=1e-2)
    assert cert is not None
    assert cert.certified and cert.f_lower >= 0.0
