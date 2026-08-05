# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Seal a certified derivative enclosure into a v1 certificate + Lean-check it.

This is "differentiation you can trust": the closed-form interval enclosure of
``sigma^(order)(z)`` (:class:`~omnibias.difference._core.extraction.DerivativeEnclosure`)
is wrapped as a **certificate-v1 interval payload**
(:func:`omnibias.core.proof.certificate.interval_certificate`), sealed
(tamper-evident digest), and handed to the Lean bridge
(:func:`omnibias.core.proof.lean_check.check_certificate`). A sign-definite
derivative (enclosure strictly ``> 0`` or ``< 0``) carries a finite, rational
obligation (``enclosed_quantity_pos`` / ``enclosed_quantity_neg``) the kernel can
re-check; a *straddling* enclosure carries no such obligation (a documented gap).

The honesty rule from the ``omnibias-dev-certificate-lean`` skill is enforced by
construction: :attr:`DerivativeProofVerdict.theorem_prover_verified` is exactly
``lean.verified``, which the bridge sets **only** on a genuine ``lake build``
pass. With no Lean toolchain it degrades gracefully (``available=False``,
``verified=False``) -- never forged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnibias.core.proof.certificate import (
    Cert,
    interval_certificate,
    verify_certificate_digest,
)
from omnibias.core.proof.lean_check import (
    LeanCheckResult,
    check_certificate,
    generate_obligation,
)
from omnibias.core.verified.interval import Interval
from omnibias.difference._core.extraction import DerivativeEnclosure


def _sign_word(value: Interval) -> str:
    """``"positive"`` / ``"negative"`` when sign-definite, else ``"indeterminate"``."""
    if value.lo > 0.0:
        return "positive"
    if value.hi < 0.0:
        return "negative"
    return "indeterminate"


def derivative_sign_certificate(enc: DerivativeEnclosure, *, meta: dict[str, Any] | None = None) -> Cert:
    """Wrap a :class:`DerivativeEnclosure` as a sealed v1 interval certificate.

    The claim records the enclosure and its sign; the honesty flags hard-wire
    ``unproven_claim=False`` (the omnibias convention). Provenance (activation, order,
    argument box, ``closed-form`` label) is carried in ``meta``.
    """
    value = enc.value
    sign = _sign_word(value)
    claim = (
        f"{enc.name}^({enc.order}) over [{enc.argument.lo!r}, {enc.argument.hi!r}] "
        f"is enclosed in [{value.lo!r}, {value.hi!r}] (sign: {sign})"
    )
    payload_meta: dict[str, Any] = {
        "name": enc.name,
        "order": enc.order,
        "argument_lo": enc.argument.lo,
        "argument_hi": enc.argument.hi,
        "label": enc.label,
        "sign": sign,
    }
    if meta:
        payload_meta.update(meta)
    return interval_certificate(claim, value, honesty={"unproven_claim": False}, meta=payload_meta)


@dataclass(frozen=True)
class DerivativeProofVerdict:
    """A sealed derivative certificate plus its Lean-kernel adjudication."""

    certificate: Cert
    sign: str
    obligation: str | None
    lean: LeanCheckResult

    @property
    def theorem_prover_verified(self) -> bool:
        """``True`` only on a genuine Lean ``lake build`` pass (never forged)."""
        return self.lean.verified

    @property
    def sealed_ok(self) -> bool:
        """Whether the certificate's tamper-evident digest matches its body."""
        return verify_certificate_digest(self.certificate)

    @property
    def obligation_generated(self) -> bool:
        """Whether a finite, Lean-checkable obligation was produced (sign-definite)."""
        return self.obligation is not None


def check_derivative_certificate(
    enc: DerivativeEnclosure,
    *,
    meta: dict[str, Any] | None = None,
    timeout: float = 600.0,
    start: Path | None = None,
) -> DerivativeProofVerdict:
    """Seal ``enc`` into a certificate and run the Lean-kernel bridge on it.

    Returns a :class:`DerivativeProofVerdict`; ``theorem_prover_verified`` is set
    only on a real kernel pass. When no Lean toolchain is present the bridge
    degrades gracefully (``lean.available is False``), so this is safe to call in
    a normal CI run.
    """
    cert = derivative_sign_certificate(enc, meta=meta)
    obligation = generate_obligation(cert)
    lean = check_certificate(cert, timeout=timeout, start=start)
    return DerivativeProofVerdict(cert, _sign_word(enc.value), obligation, lean)


__all__ = [
    "DerivativeProofVerdict",
    "check_derivative_certificate",
    "derivative_sign_certificate",
]
