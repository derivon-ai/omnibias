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

import math
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
    * **integer matrix syzygy** -- an ``integer_matrix_syzygy`` payload makes Lean
      recompute every exact matrix-vector dot product and require zero.
    * **polynomial identity over Q** -- a ``polynomial_identity_q`` payload carries
      cleared-denominator coefficient rows and strict rational margins. Lean
      independently recomputes every integer product/sum and every cross-multiplied
      positivity inequality. The producer remains responsible for enumerating the
      polynomial coefficients and Bernstein subdivision leaves.
    * **rational box-cover tiling** -- a ``box_cover_tiling`` payload carries a
      recursive axis-bisection tree. Lean reconstructs both child boxes at every
      split, checks every terminal box exactly, and checks each leaf count is at
      most the declared uniform bound.
    * **winding integer isolation** -- a ``winding_integer_isolation`` payload lifts
      both binary64 endpoints to one exact dyadic denominator and asks Lean to prove
      that the reported integer is the unique integer in the closed interval.
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
    if isinstance(cert.get("payload"), Mapping) and cert["payload"].get("type") == "realization_replay":
        from omnibias.core.proof.realization_replay import generate_replay_obligation

        return generate_replay_obligation(cert)
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

    box_cover = _extract_box_cover_tiling(cert)
    if box_cover is not None:
        parent, tree, uniform_bound = box_cover

        def rat_literal(value: tuple[int, int]) -> str:
            return f"(({value[0]} : Int), ({value[1]} : Int))"

        def box_literal(
            box: list[tuple[tuple[int, int], tuple[int, int]]],
        ) -> str:
            return "[" + ", ".join(
                f"({rat_literal(lo)}, {rat_literal(hi)})"
                for lo, hi in box
            ) + "]"

        def tree_literal(node: tuple[Any, ...]) -> str:
            if node[0] == "leaf":
                return (
                    f"(CoverTree.leaf {box_literal(node[1])} "
                    f"({node[2]} : Int))"
                )
            return (
                f"(CoverTree.split {node[1]} {rat_literal(node[2])} "
                f"{tree_literal(node[3])} {tree_literal(node[4])})"
            )

        body = (
            "/-- Exact rational intervals and recursive axis-bisection covers. -/\n"
            "abbrev RatQ := Int × Int\n"
            "abbrev RatIv := RatQ × RatQ\n\n"
            "def ratLt (a b : RatQ) : Bool :=\n"
            "  decide (0 < a.2 ∧ 0 < b.2 ∧ a.1 * b.2 < b.1 * a.2)\n\n"
            "def ratEq (a b : RatQ) : Bool :=\n"
            "  decide (0 < a.2 ∧ 0 < b.2 ∧ a.1 * b.2 = b.1 * a.2)\n\n"
            "def intervalValid (a : RatIv) : Bool := ratLt a.1 a.2\n\n"
            "def boxValid (box : List RatIv) : Bool :=\n"
            "  !box.isEmpty && box.all intervalValid\n\n"
            "def boxEq (a b : List RatIv) : Bool :=\n"
            "  a.length == b.length && (a.zip b).all (fun pair =>\n"
            "    ratEq pair.1.1 pair.2.1 && ratEq pair.1.2 pair.2.2)\n\n"
            "inductive CoverTree where\n"
            "  | leaf (box : List RatIv) (count : Int)\n"
            "  | split (axis : Nat) (cut : RatQ) (left right : CoverTree)\n\n"
            "def checkTree (bound : Int) (expected : List RatIv) : CoverTree -> Bool\n"
            "  | .leaf box count =>\n"
            "      boxValid box && boxEq expected box && decide (0 ≤ count ∧ count ≤ bound)\n"
            "  | .split axis cut left right =>\n"
            "      match expected[axis]? with\n"
            "      | none => false\n"
            "      | some interval =>\n"
            "          ratLt interval.1 cut && ratLt cut interval.2 &&\n"
            "          checkTree bound (expected.set axis (interval.1, cut)) left &&\n"
            "          checkTree bound (expected.set axis (cut, interval.2)) right\n\n"
            "set_option maxRecDepth 100000\n\n"
            "/-- Every recursive child pair exactly tiles its parent and every\n"
            "terminal count obeys the same finite bound. -/\n"
            "theorem obligation :\n"
            f"    boxValid {box_literal(parent)} &&\n"
            f"    checkTree ({uniform_bound} : Int) {box_literal(parent)} "
            f"{tree_literal(tree)} = true := by\n"
            "  decide\n"
        )
        return header + body + footer

    polynomial_identity = _extract_polynomial_identity_q(cert)
    if polynomial_identity is not None:
        equations, positive = polynomial_identity
        row_literals = ", ".join(
            "(["
            + ", ".join(f"(({left} : Int), ({right} : Int))" for left, right in terms)
            + f"], ({rhs} : Int))"
            for terms, rhs in equations
        )
        positive_literals = ", ".join(
            f"((0 : Int), (1 : Int), ({num} : Int), ({den} : Int))"
            for num, den in positive
        )
        body = (
            "/-- Cleared-denominator coefficient identities. Each pair is one\n"
            "integer product; Lean recomputes every sum independently. -/\n"
            "def productSum : List (Int × Int) -> Int\n"
            "  | [] => 0\n"
            "  | (a, b) :: rest => a * b + productSum rest\n\n"
            "def coefficientRow (row : List (Int × Int) × Int) : Bool :=\n"
            "  productSum row.1 == row.2\n\n"
            "def polynomialIdentity (rows : List (List (Int × Int) × Int)) : Bool :=\n"
            "  !rows.isEmpty && rows.all coefficientRow\n\n"
            "set_option maxRecDepth 100000\n\n"
            "/-- The identity rows and every signed Bernstein margin are exact.\n"
            "Topology and the Harnack implication remain external. -/\n"
            "theorem obligation :\n"
            f"    polynomialIdentity [{row_literals}] = true ∧\n"
            f"    allRatLt [{positive_literals}] = true := by\n"
            "  decide\n"
        )
        return stencil_header + body + footer

    matrix_syzygy = _extract_integer_matrix_syzygy(cert)
    if matrix_syzygy is not None:
        matrix, vector = matrix_syzygy
        row_literals = ", ".join(
            "[" + ", ".join(f"({value} : Int)" for value in row) + "]"
            for row in matrix
        )
        vector_literal = ", ".join(f"({value} : Int)" for value in vector)
        body = (
            "/-- Exact cleared-denominator Q[h] syzygy. Lean recomputes every\n"
            "matrix-vector product; no floating rank or Python residual is trusted. -/\n"
            "def dotInt : List Int -> List Int -> Int\n"
            "  | [], [] => 0\n"
            "  | a :: xs, b :: ys => a * b + dotInt xs ys\n"
            "  | _, _ => 1\n\n"
            "def rowSyzygy (row vector : List Int) : Bool :=\n"
            "  row.length == vector.length && dotInt row vector == 0\n\n"
            "def matrixSyzygy (matrix : List (List Int)) (vector : List Int) : Bool :=\n"
            "  !matrix.isEmpty && !vector.isEmpty && matrix.all (fun row => rowSyzygy row vector)\n\n"
            f"theorem obligation : matrixSyzygy [{row_literals}] [{vector_literal}] = true := by\n"
            "  decide\n"
        )
        return header + body + footer

    winding_isolation = _extract_winding_integer_isolation(cert)
    if winding_isolation is not None:
        lo_num, hi_num, denominator, integer = winding_isolation
        body = (
            "/-- The reported winding number is the unique integer in the exact\n"
            "dyadic lift of the outward binary64 enclosure. The analytic argument-\n"
            "principle enclosure is a trusted certificate input, not re-derived. -/\n"
            "def windingIsolated (lo hi denominator integer : Int) : Bool :=\n"
            "  decide (0 < denominator ∧\n"
            "    (integer - 1) * denominator < lo ∧\n"
            "    lo ≤ integer * denominator ∧\n"
            "    integer * denominator ≤ hi ∧\n"
            "    hi < (integer + 1) * denominator)\n\n"
            "theorem obligation :\n"
            f"    windingIsolated ({lo_num}) ({hi_num}) ({denominator}) ({integer}) = true := by\n"
            "  decide\n"
        )
        return header + body + footer

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


def _normal_rational(raw: Any) -> tuple[int, int] | None:
    """Parse and normalize a rational pair, refusing a zero denominator."""
    pair = _int_pair(raw)
    if pair is None:
        return None
    num, den = pair
    if den == 0:
        return None
    if den < 0:
        num, den = -num, -den
    return num, den






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






def _extract_integer_matrix_syzygy(
    cert: Mapping[str, Any],
) -> tuple[list[list[int]], list[int]] | None:
    """Pull a structurally valid integer matrix and proposed kernel vector."""

    payload = cert.get("payload")
    if not (
        isinstance(payload, Mapping)
        and payload.get("type") == "integer_matrix_syzygy"
    ):
        return None
    raw_matrix, raw_vector = payload.get("matrix"), payload.get("vector")
    if not (
        isinstance(raw_matrix, Sequence)
        and not isinstance(raw_matrix, str | bytes)
        and raw_matrix
        and isinstance(raw_vector, Sequence)
        and not isinstance(raw_vector, str | bytes)
        and raw_vector
        and len(raw_matrix) <= 4096
        and len(raw_vector) <= 4096
    ):
        return None
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in raw_vector
    ):
        return None
    vector = [int(value) for value in raw_vector]
    matrix: list[list[int]] = []
    for raw_row in raw_matrix:
        if not (
            isinstance(raw_row, Sequence)
            and not isinstance(raw_row, str | bytes)
            and len(raw_row) == len(vector)
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in raw_row
            )
        ):
            return None
        matrix.append([int(value) for value in raw_row])
    return matrix, vector


def _extract_polynomial_identity_q(
    cert: Mapping[str, Any],
) -> tuple[list[tuple[list[tuple[int, int]], int]], list[tuple[int, int]]] | None:
    """Pull structurally bounded coefficient rows and strict rational margins."""
    payload = cert.get("payload")
    if not (
        isinstance(payload, Mapping)
        and payload.get("type") == "polynomial_identity_q"
    ):
        return None
    raw_equations = payload.get("equations")
    raw_positive = payload.get("positive")
    if not (
        isinstance(raw_equations, Sequence)
        and not isinstance(raw_equations, str | bytes)
        and raw_equations
        and len(raw_equations) <= 4096
        and isinstance(raw_positive, Sequence)
        and not isinstance(raw_positive, str | bytes)
        and len(raw_positive) <= 262144
    ):
        return None
    equations: list[tuple[list[tuple[int, int]], int]] = []
    term_count = 0
    for raw_row in raw_equations:
        if not isinstance(raw_row, Mapping):
            return None
        raw_terms, rhs = raw_row.get("terms"), raw_row.get("rhs")
        if not (
            isinstance(raw_terms, Sequence)
            and not isinstance(raw_terms, str | bytes)
            and raw_terms
            and isinstance(rhs, int)
            and not isinstance(rhs, bool)
        ):
            return None
        terms: list[tuple[int, int]] = []
        for raw_term in raw_terms:
            if not (
                isinstance(raw_term, Sequence)
                and not isinstance(raw_term, str | bytes)
                and len(raw_term) == 2
                and all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in raw_term
                )
            ):
                return None
            terms.append((int(raw_term[0]), int(raw_term[1])))
        term_count += len(terms)
        if term_count > 262144:
            return None
        equations.append((terms, int(rhs)))
    positive: list[tuple[int, int]] = []
    for raw_value in raw_positive:
        pair = _int_pair(raw_value)
        if pair is None or pair[1] == 0:
            return None
        num, den = pair
        if den < 0:
            num, den = -num, -den
        positive.append((num, den))
    return equations, positive


def _extract_box_cover_tiling(
    cert: Mapping[str, Any],
) -> tuple[
    list[tuple[tuple[int, int], tuple[int, int]]],
    tuple[Any, ...],
    int,
] | None:
    """Pull a bounded recursive exact-Q bisection tree."""
    payload = cert.get("payload")
    if not (
        isinstance(payload, Mapping)
        and payload.get("type") == "box_cover_tiling"
    ):
        return None
    raw_parent = payload.get("parent")
    raw_tree = payload.get("tree")
    uniform_bound = payload.get("uniform_bound")
    if not (
        isinstance(uniform_bound, int)
        and not isinstance(uniform_bound, bool)
        and uniform_bound >= 0
    ):
        return None

    def parse_box(
        raw: Any,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
        if not (
            isinstance(raw, Sequence)
            and not isinstance(raw, str | bytes)
            and raw
            and len(raw) <= 256
        ):
            return None
        output: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for interval in raw:
            if not (
                isinstance(interval, Sequence)
                and not isinstance(interval, str | bytes)
                and len(interval) == 2
            ):
                return None
            lo, hi = _int_pair(interval[0]), _int_pair(interval[1])
            if lo is None or hi is None or lo[1] == 0 or hi[1] == 0:
                return None
            output.append((lo, hi))
        return output

    parent = parse_box(raw_parent)
    if parent is None:
        return None
    nodes = 0

    def parse_tree(raw: Any, depth: int = 0) -> tuple[Any, ...] | None:
        nonlocal nodes
        nodes += 1
        if nodes > 4096 or depth > 64 or not isinstance(raw, Mapping):
            return None
        kind = raw.get("kind")
        if kind == "leaf":
            box = parse_box(raw.get("box"))
            count = raw.get("count")
            if (
                box is None
                or len(box) != len(parent)
                or not isinstance(count, int)
                or isinstance(count, bool)
            ):
                return None
            return ("leaf", box, int(count))
        if kind != "split":
            return None
        axis, cut = raw.get("axis"), _int_pair(raw.get("cut"))
        if (
            not isinstance(axis, int)
            or isinstance(axis, bool)
            or not 0 <= axis < len(parent)
            or cut is None
            or cut[1] == 0
        ):
            return None
        left = parse_tree(raw.get("left"), depth + 1)
        right = parse_tree(raw.get("right"), depth + 1)
        if left is None or right is None:
            return None
        return ("split", int(axis), cut, left, right)

    tree = parse_tree(raw_tree)
    if tree is None:
        return None
    return parent, tree, int(uniform_bound)


def _float_leaf(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float.fromhex(value)
        except ValueError:
            return None
    if isinstance(value, Mapping) and set(value) == {"__f64__"}:
        tagged = value.get("__f64__")
        if isinstance(tagged, str):
            try:
                return float.fromhex(tagged)
            except ValueError:
                return None
    return None


def _extract_winding_integer_isolation(
    cert: Mapping[str, Any],
) -> tuple[int, int, int, int] | None:
    """Lift a reported binary64 winding enclosure to exact dyadic integers."""

    payload = cert.get("payload")
    if not (
        isinstance(payload, Mapping)
        and payload.get("type") == "winding_integer_isolation"
    ):
        return None
    lo, hi = _float_leaf(payload.get("lo")), _float_leaf(payload.get("hi"))
    integer = payload.get("integer")
    if (
        lo is None
        or hi is None
        or not math.isfinite(lo)
        or not math.isfinite(hi)
        or lo > hi
        or not isinstance(integer, int)
        or isinstance(integer, bool)
    ):
        return None
    lo_num, hi_num, exponent = _dyadic_triple(lo, hi)
    return lo_num, hi_num, 1 << exponent, int(integer)


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

    from omnibias.core.proof.lean_lock import generated_lean_obligation

    generated = root / _GENERATED_REL
    try:
        with generated_lean_obligation(root, generated, obligation):
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


__all__ = [
    "LeanCheckResult",
    "check_certificate",
    "generate_obligation",
    "kernel_root",
    "lean_check_available",
]
