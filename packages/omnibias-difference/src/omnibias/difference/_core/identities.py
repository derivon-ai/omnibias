# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""A Lean-checkable corpus of special-number identities (rational obligations).

Each builder turns a **finite, rational** special-number identity into a sealed
certificate-v1 payload whose obligation the Mathlib-free Lean kernel can re-check:

* :func:`bernoulli_recurrence_identity` -- the defining recurrence
  ``sum_{k=0}^{n-1} C(n,k) B_k = 0`` (``n >= 2``), an exact ``Int`` equality after
  scaling the ``B_k`` to a common denominator; a genuine cross-check that the
  Bernoulli numbers are mutually consistent;
* :func:`euler_recurrence_identity` -- the secant-number recurrence
  ``sum_{k=0}^{m} C(2m,2k) E_{2k} = 0`` (``m >= 1``);
* :func:`rational_value_identity` / :func:`zeta_negative_odd_identity` -- a
  regression equality ``computed == expected`` (e.g. ``zeta(1-2m) = -B_2m/(2m)``
  against the textbook value), certified by cross-multiplication;
* :func:`bernoulli_sign_certificate` -- the sign law ``sign(B_2m) = (-1)^{m+1}``
  that drives the Euler-Maclaurin **remainder sign**, routed through the kernel's
  ``enclosed_quantity_pos`` / ``enclosed_quantity_neg`` sign obligation.

All rational-equality builders emit the ``rational_identity`` payload discharged by
the new kernel lemma ``enclosed_quantity_eq`` (a value in the point interval
``[0, 0]`` equals ``0``). Honesty is enforced exactly as in
:mod:`omnibias.difference._core.certify`:
:attr:`RationalIdentityVerdict.theorem_prover_verified` is precisely
``lean.verified``, which the bridge sets **only** on a genuine ``lake build`` pass;
with no toolchain it degrades gracefully and the flag stays ``False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, lcm
from pathlib import Path
from typing import Any

from omnibias.core.proof.certificate import (
    Cert,
    interval_certificate,
    make_certificate,
    verify_certificate_digest,
)
from omnibias.core.proof.lean_check import (
    LeanCheckResult,
    check_certificate,
    generate_obligation,
)
from omnibias.core.verified.coeffs import bernoulli_number_exact, euler_number_exact
from omnibias.core.verified.interval import Interval

_IntTerm = tuple[int, int]


def _identity_certificate(
    claim: str, lhs_terms: list[_IntTerm], rhs: int, *, meta: dict[str, Any]
) -> Cert:
    """Seal a ``rational_identity`` payload asserting ``sum_i c_i m_i = rhs``."""
    payload = {
        "type": "rational_identity",
        "lhs_terms": [[int(c), int(m)] for c, m in lhs_terms],
        "rhs": int(rhs),
    }
    return make_certificate(claim=claim, payload=payload, honesty={"unproven_claim": False}, meta=meta)


def bernoulli_recurrence_identity(n: int) -> Cert:
    r"""Certificate for ``sum_{k=0}^{n-1} C(n,k) B_k = 0`` (``n >= 2``).

    Scales the ``B_k`` to their common denominator ``D`` so the obligation is the
    exact integer identity ``sum_k C(n,k) (B_k D) = 0``.
    """
    if n < 2:
        raise ValueError(f"Bernoulli recurrence needs n >= 2, got {n}")
    bs = [bernoulli_number_exact(k) for k in range(n)]
    den = 1
    for b in bs:
        den = lcm(den, b.denominator)
    terms: list[_IntTerm] = [(comb(n, k), int(bs[k] * den)) for k in range(n)]
    meta = {"identity": "bernoulli_recurrence", "n": n, "common_denominator": den, "label": "closed-form"}
    return _identity_certificate(f"sum_k C({n},k) B_k = 0", terms, 0, meta=meta)


def euler_recurrence_identity(m: int) -> Cert:
    r"""Certificate for ``sum_{k=0}^{m} C(2m,2k) E_{2k} = 0`` (``m >= 1``).

    The Euler (secant) numbers are integers, so no scaling is needed.
    """
    if m < 1:
        raise ValueError(f"Euler recurrence needs m >= 1, got {m}")
    terms: list[_IntTerm] = [(comb(2 * m, 2 * k), euler_number_exact(2 * k)) for k in range(m + 1)]
    meta = {"identity": "euler_recurrence", "m": m, "label": "closed-form"}
    return _identity_certificate(f"sum_k C({2 * m},2k) E_2k = 0", terms, 0, meta=meta)


def rational_value_identity(
    claim: str, computed: Fraction, expected: Fraction, *, meta: dict[str, Any] | None = None
) -> Cert:
    r"""Certificate that ``computed == expected`` via cross-multiplication.

    For ``computed = p_c/q_c`` and ``expected = p_e/q_e`` (positive denominators),
    the obligation is the exact ``Int`` equality ``p_c q_e - p_e q_c = 0``.
    """
    pc, qc = computed.numerator, computed.denominator
    pe, qe = expected.numerator, expected.denominator
    terms: list[_IntTerm] = [(qe, pc), (-qc, pe)]
    payload_meta: dict[str, Any] = {"computed": str(computed), "expected": str(expected), "label": "closed-form"}
    if meta:
        payload_meta.update(meta)
    return _identity_certificate(claim, terms, 0, meta=payload_meta)


def zeta_negative_odd_identity(m: int, expected: Fraction) -> Cert:
    r"""Certificate that ``zeta(1-2m) = -B_2m/(2m)`` equals a textbook ``expected``.

    Computes the right-hand side from the exact Bernoulli number and cross-checks it
    against the supplied literature value (e.g. ``zeta(-1) = -1/12``), so a wrong
    Bernoulli scaling or sign would be caught by the kernel.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    computed = -bernoulli_number_exact(2 * m) / Fraction(2 * m)
    meta = {"identity": "zeta_negative_odd", "m": m, "s": 1 - 2 * m}
    return rational_value_identity(f"zeta({1 - 2 * m}) = -B_{2 * m}/{2 * m}", computed, expected, meta=meta)


def bernoulli_sign_certificate(m: int) -> Cert:
    r"""Certificate that ``sign(B_2m) = (-1)^{m+1}`` (the EM remainder-sign driver).

    The Euler-Maclaurin remainder after the ``B_2m`` correction term has the sign of
    ``B_2m``; this seals the numerator (over the positive denominator) in a point
    interval so the kernel's sign obligation (``enclosed_quantity_pos`` /
    ``enclosed_quantity_neg``) certifies the alternation ``+,-,+,...``. The magnitude
    may round in ``float``, but the sign -- all that the obligation needs -- does not.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    b = bernoulli_number_exact(2 * m)
    numerator = b.numerator  # denominator > 0, so sign(B_2m) = sign(numerator)
    expected_sign = "+" if (-1) ** (m + 1) > 0 else "-"
    claim = f"sign(B_{2 * m}) = {expected_sign} (Euler-Maclaurin remainder-sign driver)"
    meta = {"identity": "bernoulli_sign", "m": m, "expected_sign": expected_sign, "label": "closed-form"}
    return interval_certificate(claim, Interval.point(float(numerator)), honesty={"unproven_claim": False}, meta=meta)


@dataclass(frozen=True)
class RationalIdentityVerdict:
    """A sealed special-number-identity certificate plus its Lean-kernel adjudication."""

    certificate: Cert
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
        """Whether a finite, Lean-checkable obligation was produced."""
        return self.obligation is not None


def check_identity_certificate(
    cert: Cert, *, timeout: float = 600.0, start: Path | None = None
) -> RationalIdentityVerdict:
    """Generate the obligation for ``cert`` and run the Lean-kernel bridge on it.

    ``theorem_prover_verified`` is set only on a real kernel pass; with no Lean
    toolchain the bridge degrades gracefully, so this is safe in a normal CI run.
    """
    obligation = generate_obligation(cert)
    lean = check_certificate(cert, timeout=timeout, start=start)
    return RationalIdentityVerdict(cert, obligation, lean)


__all__ = [
    "RationalIdentityVerdict",
    "bernoulli_recurrence_identity",
    "bernoulli_sign_certificate",
    "check_identity_certificate",
    "euler_recurrence_identity",
    "rational_value_identity",
    "zeta_negative_odd_identity",
]
