# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Deterministic driver for the Mathlib-backed formal loop.

:func:`drive_obligation` is the small, reproducible primitive the
``omnibias-dev-formal-agent`` skill leans on.  It turns a certificate into one
*actionable* pass of the loop: classify the finite obligation, drive
:func:`~omnibias.formal.mathlib_check.check_certificate` (which re-derives the
obligation exactly over ``ℚ``, emits Lean, and runs ``lake build`` -- the
``generate -> lake build -> report`` cycle), then distil the outcome into a
:class:`DriveReport` carrying a short, rule-based ``next_action``.

It is deliberately **deterministic plumbing, not an autonomous prover**.  The
only non-pure step is the single ``lake`` invocation delegated to the bridge, and
it inherits every honesty invariant of that bridge: the Mathlib trust tier
(:data:`~omnibias.formal.mathlib_check.MATHLIB_CLAIM_KEY`, ``"mathlib_verified"``)
is reported **only** on a genuine ``lake`` pass, is never conflated with the
Mathlib-free kernel's ``theorem_prover_verified``, and never implies
``unproven_claim``.  With no Lean toolchain present the driver degrades gracefully
(``available=False``) and its ``next_action`` says how to get a real verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnibias.formal.mathlib_check import (
    MATHLIB_CLAIM_KEY,
    check_certificate,
    classify_obligation,
)

#: Substrings marking the salient lines of a failed ``lake build`` log.
_ERROR_MARKERS = ("error", "sorry", ".lean:")
#: How many salient lines to keep in a distilled failure summary.
_MAX_FAILURE_LINES = 3


@dataclass(frozen=True)
class DriveReport:
    """The outcome of one deterministic pass of the formal loop.

    ``tier`` is the earned trust tier -- :data:`MATHLIB_CLAIM_KEY` on a genuine
    ``lake`` pass, else ``None``.  It is never ``theorem_prover_verified`` and
    never implies ``unproven_claim``.  ``next_action`` is deterministic, rule-based
    guidance for the agent driving the loop.
    """

    obligation_class: str | None
    attempted: bool
    available: bool
    verified: bool
    tier: str | None
    failure: str | None
    obligation: str
    detail: str
    next_action: str


def _summarize_lake_failure(detail: str) -> str:
    """Distil a ``lake build`` log tail to its first few salient lines.

    Keeps the first :data:`_MAX_FAILURE_LINES` lines that mention an ``error`` /
    ``sorry`` / a ``.lean:`` location, joined by ``" | "``; falls back to the last
    non-empty line so a failure is never reported as empty.
    """
    lines = [line.strip() for line in detail.splitlines() if line.strip()]
    salient = [line for line in lines if any(marker in line.lower() for marker in _ERROR_MARKERS)]
    if salient:
        return " | ".join(salient[:_MAX_FAILURE_LINES])
    return lines[-1] if lines else detail.strip()


def drive_obligation(
    cert: Mapping[str, Any],
    *,
    start: Path | None = None,
    timeout: float = 1800.0,
) -> DriveReport:
    """Run one deterministic pass of the formal loop over ``cert``.

    Classifies the certificate's finite obligation, drives
    :func:`~omnibias.formal.mathlib_check.check_certificate`, and reports an
    actionable :class:`DriveReport`.  Raises nothing: an unsupported payload, a
    missing toolchain, and a failing build are each reported with a
    ``next_action``.  ``tier`` is :data:`MATHLIB_CLAIM_KEY` only on a genuine
    ``lake`` pass; ``failure`` is populated only when Mathlib rejected the build.
    """
    obligation_class = classify_obligation(cert)
    result = check_certificate(cert, timeout=timeout, start=start)
    attempted = obligation_class is not None

    verified = result.verified
    available = result.available
    tier = MATHLIB_CLAIM_KEY if verified else None
    failure: str | None = None

    if not attempted:
        next_action = (
            "No Mathlib-checkable finite obligation in this certificate. Either the "
            "payload's obligation class is unsupported -- add a Check/* capability "
            "lemma plus a generator in mathlib_check.py -- or discharge it with the "
            "Mathlib-free kernel (omnibias.core.proof.lean_check) instead."
        )
    elif not available:
        next_action = (
            "Obligation generated, but no Lean toolchain / analytic checkout is "
            "present. Run under the non-blocking lean-analytic workflow, or install "
            "elan + lake and build formal/omnibias-analytic, to get a real verdict."
        )
    elif verified:
        next_action = (
            "Verified by the Mathlib kernel. Attach the mathlib_verified tier only "
            "(via evaluate_with_mathlib) -- never theorem_prover_verified, never "
            "unproven_claim."
        )
    else:
        failure = _summarize_lake_failure(result.detail)
        next_action = (
            "Mathlib rejected the obligation. The emitted obligation is the source "
            "of truth: fix the certificate's payload or add / repair a sorry-free "
            "Check/* lemma. Never hand-edit Generated.lean (the bridge overwrites it)."
        )

    return DriveReport(
        obligation_class=obligation_class,
        attempted=attempted,
        available=available,
        verified=verified,
        tier=tier,
        failure=failure,
        obligation=result.obligation,
        detail=result.detail,
        next_action=next_action,
    )


__all__ = [
    "DriveReport",
    "drive_obligation",
]
