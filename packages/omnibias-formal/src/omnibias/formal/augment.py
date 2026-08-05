# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Wire the Mathlib trust tier into a :class:`~omnibias.core.proof.Verdict`.

:func:`evaluate_with_mathlib` runs the ordinary :meth:`ProofMachine.evaluate`
pipeline and then, on a ``PROVED`` / ``DISPROVED`` verdict with a certificate,
re-checks the certificate's finite obligation with the Mathlib-backed bridge
(:mod:`omnibias.formal.mathlib_check`).  The outcome is reported on a **separate,
larger** trust tier, :data:`~omnibias.formal.mathlib_check.MATHLIB_CLAIM_KEY`
(``"mathlib_verified"``), that is never conflated with the minimal Mathlib-free
kernel's ``theorem_prover_verified`` and never implies ``unproven_claim``.

Why this lives outside ``omnibias.core``.  The core :class:`Verdict` is frozen and
core must not depend on this optional package, so the tier is attached by a
consumer-side wrapper (:class:`MathlibVerdict`) rather than a new core field.

The honesty gate mirrors core's ``theorem_prover_verified`` gate: if a conjecture
*asserts* the ``mathlib_verified`` claim but Mathlib does not verify the
obligation, a ``PROVED`` verdict is downgraded to ``BLOCKED``.  Core's generic
:func:`~omnibias.core.proof.honesty_gate` only special-cases *its* reserved key
(``theorem_prover_verified``), so this wrapper first **strips** the reserved
``mathlib_verified`` key from the conjecture it hands to core -- exactly the way
core ignores its own formal key -- and then adjudicates the tier itself.  This is
the only correct way to add the tier without editing core.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from omnibias.core.proof import Conjecture, ProofMachine, Verdict
from omnibias.formal.mathlib_check import MATHLIB_CLAIM_KEY, check_certificate


@dataclasses.dataclass(frozen=True)
class MathlibVerdict:
    """A core :class:`Verdict` augmented with the Mathlib trust tier.

    ``verdict`` is the (possibly downgraded) core adjudication; the ``mathlib_*``
    fields record the Mathlib-backed bridge outcome.  ``mathlib_verified`` is set
    **only** on a genuine ``lake`` pass and is a distinct, larger trust base than
    ``verdict.theorem_prover_verified`` -- the two are never conflated, and neither
    ever implies ``unproven_claim``.
    """

    verdict: Verdict
    mathlib_verified: bool
    mathlib_available: bool
    mathlib_obligation: str = ""
    mathlib_detail: str = ""

    @property
    def status(self) -> str:
        return self.verdict.status

    @property
    def obligations(self) -> tuple[str, ...]:
        return self.verdict.obligations

    @property
    def proved(self) -> bool:
        return self.verdict.proved

    @property
    def blocked(self) -> bool:
        return self.verdict.blocked

    def summary(self) -> dict[str, Any]:
        """The core verdict digest plus the Mathlib tier (never claims an unproven result)."""
        digest = self.verdict.summary()
        digest["mathlib_verified"] = self.mathlib_verified
        digest["mathlib_available"] = self.mathlib_available
        return digest


def evaluate_with_mathlib(
    machine: ProofMachine,
    conjecture: Conjecture,
    *,
    replay: bool = True,
    lean_check: bool = False,
    strict: bool = False,
    mathlib_check: bool = False,
    start: Path | None = None,
) -> MathlibVerdict:
    """Evaluate ``conjecture`` and attach the Mathlib ``mathlib_verified`` tier.

    The Mathlib bridge runs when either ``mathlib_check`` is set or the conjecture
    asserts the ``mathlib_verified`` claim, and only on a ``PROVED`` / ``DISPROVED``
    verdict that carries a certificate.  Asserting the claim without a passing
    Mathlib build downgrades a ``PROVED`` verdict to ``BLOCKED`` (the honesty gate);
    ``mathlib_check`` alone records the tier without gating (mirroring core's
    ``lean_check``).  ``theorem_prover_verified`` is never set or read here.
    """
    claims_mathlib = bool(conjecture.claims.get(MATHLIB_CLAIM_KEY, False))

    # Strip the reserved consumer key before core sees it, so core's generic
    # honesty gate does not treat "mathlib_verified" as a certificate-backed flag
    # (core only special-cases its own reserved theorem_prover_verified key).
    core_conjecture = conjecture
    if MATHLIB_CLAIM_KEY in conjecture.claims:
        stripped = {k: v for k, v in conjecture.claims.items() if k != MATHLIB_CLAIM_KEY}
        core_conjecture = dataclasses.replace(conjecture, claims=stripped)

    verdict = machine.evaluate(
        core_conjecture, replay=replay, lean_check=lean_check, strict=strict
    )

    verified = False
    available = False
    obligation = ""
    detail = ""
    if (
        (mathlib_check or claims_mathlib)
        and verdict.certificate is not None
        and verdict.status in ("PROVED", "DISPROVED")
    ):
        result = check_certificate(verdict.certificate, start=start)
        verified, available = result.verified, result.available
        obligation, detail = result.obligation, result.detail

    if verdict.status == "PROVED" and claims_mathlib and not verified:
        verdict = dataclasses.replace(
            verdict,
            status="BLOCKED",
            honesty_ok=False,
            obligations=verdict.obligations
            + (
                "honesty gate: 'mathlib_verified' asserted but Mathlib did not verify "
                "the certificate's finite obligation",
            ),
        )

    return MathlibVerdict(
        verdict=verdict,
        mathlib_verified=verified,
        mathlib_available=available,
        mathlib_obligation=obligation,
        mathlib_detail=detail,
    )


__all__ = [
    "MathlibVerdict",
    "evaluate_with_mathlib",
]
