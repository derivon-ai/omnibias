# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The Lean-kernel bridge: turn a certificate's finite obligation into a theorem.

This runner connects the certificate format to the Lean kernel
(``formal/omnibias-verified-kernel``).  It

1. extracts the **finite, rational** obligation carried by a certificate
   (spectral-gap positivity, or the sign of an enclosed quantity), refusing any
   certificate whose ``digest`` does not match its body (tamper-evidence);
2. emits a tiny Lean source file (``Omnibias/Generated.lean``) that discharges the
   obligation by chaining the kernel's *proven* soundness lemmas;
3. invokes ``lake build`` so the **Lean kernel** re-checks it; and
4. reports whether the kernel accepted the proof.

It is deliberately dependency-free (standard library only) so it can live in
``omnibias.core``.  When no Lean toolchain (``lake``) or kernel checkout is
present it degrades gracefully -- :func:`lean_check_available` returns ``False``
and :func:`check_certificate` returns an ``available=False`` result rather than
raising -- so a normal test / CI run without Lean is unaffected.  Only a genuine
``lake`` pass yields ``verified=True``; the flag can never be forged by the
certificate itself.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from omnibias.core.proof.certificate import verify_certificate_digest

#: Relative location of the Lean kernel within the repository.
_KERNEL_REL = Path("formal") / "omnibias-verified-kernel"
#: The generated-obligation module the bridge overwrites (then restores).
_GENERATED_REL = Path("Omnibias") / "Generated.lean"


@dataclass(frozen=True)
class LeanCheckResult:
    """The outcome of a Lean-kernel check of a single certificate."""

    verified: bool
    available: bool
    obligation: str
    detail: str


def kernel_root(start: Path | None = None) -> Path | None:
    """Locate ``formal/omnibias-verified-kernel`` by walking up from ``start``."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / _KERNEL_REL
        if (candidate / "lakefile.lean").is_file():
            return candidate
    return None


def lean_check_available(start: Path | None = None) -> bool:
    """``True`` iff both a ``lake`` executable and the kernel checkout are present."""
    return shutil.which("lake") is not None and kernel_root(start) is not None


# --------------------------------------------------------------------------- #
# Obligation extraction -> Lean source.
# --------------------------------------------------------------------------- #
def _scaled_pair(lo: float, hi: float) -> tuple[int, int]:
    """Scale a rational interval ``[lo, hi]`` to integers over a common denominator."""
    flo, fhi = Fraction(lo), Fraction(hi)
    den = _lcm(flo.denominator, fhi.denominator)
    return flo.numerator * (den // flo.denominator), fhi.numerator * (den // fhi.denominator)


def _lcm(a: int, b: int) -> int:
    from math import gcd

    return abs(a * b) // gcd(a, b) if a and b else (a or b or 1)


def _dyadic_triple(lo: float, hi: float) -> tuple[int, int, int]:
    """``(loNum, hiNum, exp)`` with ``lo = loNum / 2**exp``, ``hi = hiNum / 2**exp``.

    Every finite Python ``float`` is an exact dyadic rational (an integer
    numerator over a power-of-two denominator, via
    :meth:`float.as_integer_ratio`), so this loses no information -- unlike
    :func:`_scaled_pair`'s generic rational LCM, no rounding or
    approximation is involved, just lifting the coarser endpoint to the
    finer of the two exponents.
    """
    lo_num, lo_den = float(lo).as_integer_ratio()
    hi_num, hi_den = float(hi).as_integer_ratio()
    lo_exp = lo_den.bit_length() - 1
    hi_exp = hi_den.bit_length() - 1
    exp = max(lo_exp, hi_exp)
    lo_num *= 1 << (exp - lo_exp)
    hi_num *= 1 << (exp - hi_exp)
    return lo_num, hi_num, exp


def generate_obligation(cert: Mapping[str, Any]) -> str | None:
    r"""Return Lean source discharging ``cert``'s finite obligation, or ``None``.

    Supported obligations:

    * **spectral gap** -- a certificate carrying ``subdominant_ratio_upper`` ``< 1``
      yields ``0 < gapNumerator rn rd`` via ``spectral_gap_pos``;
    * **positive-definite matrix** -- a v1 ``positive_definite`` payload (interval
      ``LDL^T`` pivots) yields the inertia-vector obligation ``allPivotsPos [...] =
      true`` (``matrix_positive_definite_certified``) when every pivot's lower
      endpoint is positive -- the full PD statement, not a single scalar;
    * **enclosed-quantity sign** -- a v1 ``interval`` payload (or any mapping with a
      ``lo``/``hi`` interval) yields ``enclosed_quantity_pos`` (when ``lo > 0``) or
      ``enclosed_quantity_neg`` (when ``hi < 0``).
    * **rational identity** -- a ``rational_identity`` payload carrying integer
      ``lhs_terms`` ``[[c, m], ...]`` and an integer ``rhs`` yields the exact ``Int``
      equality ``sum_i c_i m_i = rhs`` via ``enclosed_quantity_eq`` (the difference
      lies in the point interval ``[0, 0]``).  This is how a special-number identity
      (a Bernoulli recurrence, ``zeta(1-2m) = -B_2m/(2m)``, ...) -- scaled to a
      common ``Int`` denominator -- earns a kernel-checked *equality*, not a sign.
    * **PDE finite margin** -- a ``pinn_aposteriori_error`` payload may carry
      ``finite_obligation.margin = threshold - error_bound``.  The kernel checks
      only the finite inequality, not the analytic PDE theorem.
    * **stencil consistency** -- a ``rational_stencil_consistency`` payload of
      ``(lhs, rhs)`` rational pairs (moments ``C_j`` and the reported ``C_N``)
      yields ``allRatEq [...] = true`` via ``Omnibias.RationalStencil``.
    * **stencil poisedness** -- a ``rational_poisedness`` payload yields
      ``allIntGe`` (Polya) and ``ratNez`` (nonzero determinant). Neither
      payload states a collapse or a remainder over a function class.
    * **interval replay trace** -- an ``interval_replay_trace`` payload
      (:meth:`omnibias.core.proof.replay.ReplayTrace.to_payload`) yields
      ``Omnibias.Replay.replayOk [...] = true`` via ``decide``: the Lean
      kernel genuinely re-executes every recorded ``sub``/``mul`` step
      against its own proven ``ZInterval`` arithmetic and checks each
      recorded envelope contains the exact recomputation. Scoped to
      exactly the ``literal``/``sub``/``mul`` vocabulary
      ``Omnibias/Replay.lean`` implements; emitted unconditionally (no
      Python-side truth pre-check), since the entire point is that Lean is
      the independent arbiter, not a rubber stamp of a Python conclusion.
    * **domain-subdivision coverage** -- a ``domain_subdivision`` payload
      (:meth:`omnibias.core.proof.replay.DomainSubdivisionCertificate.to_payload`)
      yields ``Omnibias.Subdivision.coversNoGaps ... = true`` via
      ``decide``, checking the visited leaves tile the claimed domain
      edge-to-edge. Only emitted when the Python-side
      ``covers_no_gaps()`` check already agrees (matching the existing
      positive-definite-pivot / poisedness pattern of gating on a
      Python-side truth check before asking Lean to re-derive it); a
      certificate whose leaves have a gap is refused here, before any Lean
      is emitted.
    """
    header = (
        "/- AUTO-GENERATED by omnibias.core.proof.lean_check. DO NOT EDIT. -/\n"
        "import Omnibias.Certificate\n\n"
        "namespace Omnibias.Generated\n\n"
    )
    stencil_header = (
        "/- AUTO-GENERATED by omnibias.core.proof.lean_check. DO NOT EDIT. -/\n"
        "import Omnibias.RationalStencil\n\n"
        "namespace Omnibias.Generated\n\n"
    )
    footer = "\nend Omnibias.Generated\n"

    stencil_pairs = _extract_stencil_pairs(cert)
    if stencil_pairs is not None:
        lits = ", ".join(
            f"(({p}), {q}, {r}, {s})" for p, q, r, s in stencil_pairs
        )
        body = (
            "/-- Finite stencil identities: each (p/q) = (r/s) is the Int\n"
            "cross-multiplication p*s = r*q with nonzero denominators. Algebra\n"
            "only; no collapse and no remainder over a function class. -/\n"
            f"theorem obligation : allRatEq [{lits}] = true := by decide\n"
        )
        return stencil_header + body + footer

    replay_steps = _extract_replay_trace(cert)
    if replay_steps is not None:
        lits = ", ".join(
            f"⟨{kind}, {a0}, {a1}, {lo}, {hi}, {exp}⟩" for kind, a0, a1, lo, hi, exp in replay_steps
        )
        replay_header = (
            "/- AUTO-GENERATED by omnibias.core.proof.lean_check. DO NOT EDIT. -/\n"
            "import Omnibias.Replay\n\n"
            "namespace Omnibias.Generated\n\n"
        )
        body = (
            "/-- Finite straight-line replay: every recorded `sub`/`mul` step's\n"
            "envelope genuinely contains the exact recomputation from prior\n"
            "recorded steps, via the kernel's proven `ZInterval` arithmetic.\n"
            "`literal` steps are trusted inputs, not re-derived. -/\n"
            f"theorem obligation : Omnibias.Replay.replayOk [{lits}] = true := by decide\n"
        )
        return replay_header + body + footer

    subdivision = _extract_domain_subdivision(cert)
    if subdivision is not None:
        resolution, leaves = subdivision
        lits = ", ".join(f"⟨{lo}, {hi}⟩" for lo, hi in leaves)
        subdivision_header = (
            "/- AUTO-GENERATED by omnibias.core.proof.lean_check. DO NOT EDIT. -/\n"
            "import Omnibias.Subdivision\n\n"
            "namespace Omnibias.Generated\n\n"
        )
        body = (
            "/-- Finite domain-subdivision coverage: the visited leaves tile\n"
            f"[0, {resolution}] edge-to-edge, with no gap and no overlap. Says\n"
            "nothing about any individual leaf's own analytic conclusion. -/\n"
            f"theorem obligation : "
            f"Omnibias.Subdivision.coversNoGaps ({resolution}) [{lits}] = true := by decide\n"
        )
        return subdivision_header + body + footer

    poised = _extract_poisedness(cert)
    if poised is not None:
        polya, det_n, det_d = poised
        polya_lits = ", ".join(f"(({have}), {need})" for have, need in polya)
        body = (
            "/-- Finite poisedness: Polya integer comparisons and a nonzero\n"
            "rational determinant witness. Algebra only. -/\n"
            f"theorem obligation : allIntGe [{polya_lits}] = true ∧ "
            f"ratNez ({det_n}) {det_d} = true := by decide\n"
        )
        return stencil_header + body + footer

    pivots = _extract_pd_pivots(cert)
    if pivots is not None:
        scaled = [_scaled_pair(lo, hi) for lo, hi in pivots]
        if scaled and all(ilo > 0 for ilo, _ihi in scaled):
            pivot_lits = ", ".join(f"⟨{ilo}, {ihi}⟩" for ilo, ihi in scaled)
            pd_header = (
                "/- AUTO-GENERATED by omnibias.core.proof.lean_check. DO NOT EDIT. -/\n"
                "import Omnibias.LDLT\n\n"
                "namespace Omnibias.Generated\n\n"
            )
            body = (
                "/-- Certified positive-definite LDLᵀ inertia: every pivot interval is\n"
                "strictly positive, so the negative inertia is zero -- the matrix box is\n"
                "positive definite (factorisation a trusted Python input). -/\n"
                f"theorem obligation : allPivotsPos [{pivot_lits}] = true := by decide\n"
            )
            return pd_header + body + footer

    identity = _extract_rational_identity(cert)
    if identity is not None:
        lhs_terms, rhs = identity
        lhs = " + ".join(f"({c}) * ({m})" for c, m in lhs_terms)
        body = (
            f"/-- Rational special-number identity: sum_i c_i m_i = {rhs} (integers over a\n"
            f"common denominator), certified as an exact Int equality via the point interval\n"
            f"[0, 0] and `enclosed_quantity_eq`. -/\n"
            f"theorem obligation : ({lhs} : Int) = {rhs} := by\n"
            f"  have hx : ZInterval.Mem (({lhs}) - ({rhs})) ⟨0, 0⟩ := by\n"
            f"    simp only [ZInterval.Mem]; omega\n"
            f"  have key : ({lhs}) - ({rhs}) = 0 := ZInterval.eq_of_mem_point hx (by decide)\n"
            f"  omega\n"
        )
        return header + body + footer

    ratio = cert.get("subdominant_ratio_upper")
    if isinstance(ratio, int | float) and 0.0 <= float(ratio) < 1.0:
        frac = Fraction(float(ratio))
        rn, rd = frac.numerator, frac.denominator
        if rd <= 0:
            rn, rd = -rn, -rd
        body = (
            f"/-- Spectral-gap positivity from subdominant-ratio upper bound "
            f"{rn}/{rd} < 1. -/\n"
            f"theorem obligation : 0 < gapNumerator {rn} {rd} :=\n"
            f"  spectral_gap_pos (by decide) (by decide)\n"
        )
        return header + body + footer

    interval = _extract_interval(cert)
    if interval is not None:
        lo, hi = interval
        ilo, ihi = _scaled_pair(lo, hi)
        if ilo > 0:
            body = (
                f"/-- Enclosed quantity is positive: x in [{ilo}, {ihi}] => 0 < x. -/\n"
                f"theorem obligation (x : Int) "
                f"(hx : ZInterval.Mem x ⟨{ilo}, {ihi}⟩) : 0 < x :=\n"
                f"  enclosed_quantity_pos hx (by decide)\n"
            )
            return header + body + footer
        if ihi < 0:
            body = (
                f"/-- Enclosed quantity is negative: x in [{ilo}, {ihi}] => x < 0. -/\n"
                f"theorem obligation (x : Int) "
                f"(hx : ZInterval.Mem x ⟨{ilo}, {ihi}⟩) : x < 0 :=\n"
                f"  enclosed_quantity_neg hx (by decide)\n"
            )
            return header + body + footer
    return None


def _extract_pd_pivots(cert: Mapping[str, Any]) -> list[tuple[float, float]] | None:
    """Pull the interval ``LDL^T`` pivot list from a v1 ``positive_definite`` payload."""
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "positive_definite"):
        return None
    pivots = payload.get("pivots")
    if not (isinstance(pivots, Sequence) and not isinstance(pivots, str | bytes) and pivots):
        return None
    out: list[tuple[float, float]] = []
    for p in pivots:
        if not (isinstance(p, Mapping) and "lo" in p and "hi" in p):
            return None
        lo, hi = p["lo"], p["hi"]
        out.append(
            (float.fromhex(lo), float.fromhex(hi)) if isinstance(lo, str) else (float(lo), float(hi))
        )
    return out


def _extract_replay_trace(
    cert: Mapping[str, Any],
) -> list[tuple[int, int, int, int, int, int]] | None:
    """Pull ``(kind, arg0, arg1, recLo, recHi, recExp)`` sextuples from an
    ``interval_replay_trace`` payload, one per step, in trace order
    (``kind``: ``0`` = literal, ``1`` = sub, ``2`` = mul -- matching
    ``Omnibias.Replay.Step``).

    Structural validation only (well-formed shape, in-range operand
    indices for ``sub``/``mul``): this does **not** check that any step's
    recorded envelope is arithmetically correct -- that is exactly what the
    emitted ``Omnibias.Replay.replayOk ... := by decide`` obligation
    checks, genuinely, in the Lean kernel.
    """
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "interval_replay_trace"):
        return None
    raw_steps = payload.get("steps")
    if not (isinstance(raw_steps, Sequence) and not isinstance(raw_steps, str | bytes) and raw_steps):
        return None
    kind_of = {"literal": 0, "sub": 1, "mul": 2}
    out: list[tuple[int, int, int, int, int, int]] = []
    for i, raw in enumerate(raw_steps):
        if not isinstance(raw, Mapping):
            return None
        op = raw.get("op")
        if op not in kind_of:
            return None
        args = raw.get("args")
        if not (isinstance(args, Sequence) and not isinstance(args, str | bytes)):
            return None
        want = 0 if op == "literal" else 2
        if len(args) != want:
            return None
        try:
            arg0 = int(args[0]) if len(args) > 0 else 0
            arg1 = int(args[1]) if len(args) > 1 else 0
        except (TypeError, ValueError):
            return None
        if op != "literal" and not (0 <= arg0 < i and 0 <= arg1 < i):
            return None
        lo_raw, hi_raw = raw.get("lo"), raw.get("hi")
        if not (isinstance(lo_raw, str) and isinstance(hi_raw, str)):
            return None
        try:
            lo, hi = float.fromhex(lo_raw), float.fromhex(hi_raw)
        except ValueError:
            return None
        rec_lo, rec_hi, rec_exp = _dyadic_triple(lo, hi)
        out.append((kind_of[op], arg0, arg1, rec_lo, rec_hi, rec_exp))
    return out


def _extract_domain_subdivision(
    cert: Mapping[str, Any],
) -> tuple[int, list[tuple[int, int]]] | None:
    """Pull ``(resolution, leaves)`` from a ``domain_subdivision`` payload.

    Requires the visited leaves to already tile ``[0, resolution]`` with no
    gap -- the same combinatorial check
    ``omnibias.core.proof.replay.DomainSubdivisionCertificate.covers_no_gaps``
    and ``Omnibias.Subdivision.coversNoGaps`` independently re-derive.  A
    certificate with a genuine gap is refused *here*, before any Lean is
    emitted, mirroring how the positive-definite-pivot obligation above
    only fires when every pivot is already known positive.
    """
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "domain_subdivision"):
        return None
    resolution = payload.get("resolution")
    raw_leaves = payload.get("leaves")
    if not (
        isinstance(resolution, int)
        and not isinstance(resolution, bool)
        and resolution > 0
        and isinstance(raw_leaves, Sequence)
        and not isinstance(raw_leaves, str | bytes)
        and raw_leaves
    ):
        return None
    leaves: list[tuple[int, int]] = []
    for raw in raw_leaves:
        if not isinstance(raw, Mapping):
            return None
        lo, hi = raw.get("lo"), raw.get("hi")
        if not (
            isinstance(lo, int)
            and not isinstance(lo, bool)
            and isinstance(hi, int)
            and not isinstance(hi, bool)
        ):
            return None
        leaves.append((lo, hi))
    if leaves[0][0] != 0:
        return None
    prev_hi = leaves[0][0]
    for lo, hi in leaves:
        if lo != prev_hi or lo >= hi:
            return None
        prev_hi = hi
    if prev_hi != resolution:
        return None
    return resolution, leaves


def _int_pair(raw: Any) -> tuple[int, int] | None:
    """Parse a ``[num, den]`` pair stored as ints or decimal-free strings."""
    if not (isinstance(raw, Sequence) and not isinstance(raw, str | bytes) and len(raw) == 2):
        return None
    try:
        num, den = int(raw[0]), int(raw[1])
    except (TypeError, ValueError):
        return None
    return num, den


def _extract_stencil_pairs(
    cert: Mapping[str, Any],
) -> list[tuple[int, int, int, int]] | None:
    """Pull ``(p, q, r, s)`` cross-multiply tuples from a stencil payload."""
    payload = cert.get("payload")
    if not (
        isinstance(payload, Mapping)
        and payload.get("type") == "rational_stencil_consistency"
    ):
        return None
    raw = payload.get("conditions")
    if not (isinstance(raw, Sequence) and not isinstance(raw, str | bytes) and raw):
        return None
    pairs: list[tuple[int, int, int, int]] = []
    for row in raw:
        if not isinstance(row, Mapping):
            return None
        lhs = _int_pair(row.get("lhs"))
        rhs = _int_pair(row.get("rhs"))
        if lhs is None or rhs is None:
            return None
        pairs.append((lhs[0], lhs[1], rhs[0], rhs[1]))
    lead_lhs = _int_pair(payload.get("leading_lhs"))
    lead_rhs = _int_pair(payload.get("leading_coeff"))
    if lead_lhs is None or lead_rhs is None:
        return None
    pairs.append((lead_lhs[0], lead_lhs[1], lead_rhs[0], lead_rhs[1]))
    return pairs


def _extract_poisedness(
    cert: Mapping[str, Any],
) -> tuple[list[tuple[int, int]], int, int] | None:
    """Pull Polya ``(have, need)`` pairs and a determinant ``n/d``."""
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "rational_poisedness"):
        return None
    raw = payload.get("polya")
    det = _int_pair(payload.get("det"))
    if not (
        isinstance(raw, Sequence)
        and not isinstance(raw, str | bytes)
        and raw
        and det is not None
    ):
        return None
    polya: list[tuple[int, int]] = []
    for row in raw:
        if not isinstance(row, Mapping):
            return None
        have, need = row.get("have"), row.get("need")
        if not (
            isinstance(have, int)
            and not isinstance(have, bool)
            and isinstance(need, int)
            and not isinstance(need, bool)
        ):
            return None
        polya.append((int(have), int(need)))
    return polya, det[0], det[1]


def _extract_rational_identity(
    cert: Mapping[str, Any],
) -> tuple[list[tuple[int, int]], int] | None:
    """Pull ``(lhs_terms, rhs)`` integer data from a ``rational_identity`` payload."""
    payload = cert.get("payload")
    if not (isinstance(payload, Mapping) and payload.get("type") == "rational_identity"):
        return None
    raw_terms = payload.get("lhs_terms")
    rhs = payload.get("rhs")
    if not (
        isinstance(raw_terms, Sequence)
        and not isinstance(raw_terms, str | bytes)
        and raw_terms
        and isinstance(rhs, int)
        and not isinstance(rhs, bool)
    ):
        return None
    terms: list[tuple[int, int]] = []
    for pair in raw_terms:
        if not (
            isinstance(pair, Sequence)
            and not isinstance(pair, str | bytes)
            and len(pair) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in pair)
        ):
            return None
        terms.append((int(pair[0]), int(pair[1])))
    return terms, int(rhs)


def _extract_interval(cert: Mapping[str, Any]) -> tuple[float, float] | None:
    """Pull an ``[lo, hi]`` enclosure from a v1 interval payload or a raw mapping."""
    payload = cert.get("payload")
    if isinstance(payload, Mapping) and payload.get("type") == "interval":
        iv = payload.get("interval")
        if isinstance(iv, Mapping) and "lo" in iv and "hi" in iv:
            return float.fromhex(iv["lo"]), float.fromhex(iv["hi"])
    if isinstance(payload, Mapping) and payload.get("type") == "pinn_aposteriori_error":
        finite = payload.get("finite_obligation")
        if isinstance(finite, Mapping) and finite.get("type") == "error_bound_le_threshold":
            margin = finite.get("margin")
            if (
                isinstance(margin, Sequence)
                and not isinstance(margin, str | bytes)
                and len(margin) == 2
            ):
                return float(margin[0]), float(margin[1])
    iv = cert.get("interval")
    if isinstance(iv, Mapping) and "lo" in iv and "hi" in iv:
        lo, hi = iv["lo"], iv["hi"]
        if isinstance(lo, str):
            return float.fromhex(lo), float.fromhex(hi)
        return float(lo), float(hi)
    return None


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def check_certificate(
    cert: Mapping[str, Any],
    *,
    timeout: float = 600.0,
    start: Path | None = None,
) -> LeanCheckResult:
    """Generate the obligation, run ``lake build``, and report the kernel verdict.

    The certificate must carry a **valid seal**: an absent or mismatched ``digest``
    is rejected before any Lean is emitted.  Accepting an unsealed payload here
    would let an unsigned mapping earn ``verified=True``, defeating the
    tamper-evidence the seal exists to provide -- so a missing digest is refused
    exactly like a forged one.  If the toolchain is unavailable the result has
    ``available=False`` and ``verified=False``.
    """
    if not verify_certificate_digest(cert):
        reason = (
            "certificate digest mismatch (tampered/stale)"
            if "digest" in cert
            else "unsealed certificate (no digest); seal it with seal_certificate() first"
        )
        return LeanCheckResult(False, True, "", reason)

    obligation = generate_obligation(cert)
    if obligation is None:
        return LeanCheckResult(False, True, "", "no finite Lean-checkable obligation in certificate")

    root = kernel_root(start)
    if root is None or shutil.which("lake") is None:
        return LeanCheckResult(False, False, obligation, "Lean toolchain or kernel checkout unavailable")

    generated = root / _GENERATED_REL
    original = generated.read_text(encoding="utf-8") if generated.exists() else None
    try:
        generated.write_text(obligation, encoding="utf-8")
        proc = subprocess.run(
            ["lake", "build"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        ok = proc.returncode == 0
        detail = "Lean kernel accepted the obligation" if ok else (proc.stderr or proc.stdout)[-2000:]
        return LeanCheckResult(ok, True, obligation, detail)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
        return LeanCheckResult(False, False, obligation, f"lake invocation failed: {exc}")
    finally:
        if original is not None:
            generated.write_text(original, encoding="utf-8")


__all__ = [
    "LeanCheckResult",
    "check_certificate",
    "generate_obligation",
    "kernel_root",
    "lean_check_available",
]
