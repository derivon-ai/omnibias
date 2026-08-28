# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Finite straight-line replay traces + domain-subdivision coverage
(phase1-lean-replay).

Two small, additive, **opt-in** capabilities that let the Lean kernel
(``formal/omnibias-verified-kernel``) genuinely re-derive a piece of an
:class:`~omnibias.core.verified.interval.Interval` computation, instead of
merely trusting a Python-reported conclusion:

* :class:`ReplayTrace` (built by :class:`ReplayRecorder`) -- an ordered,
  serializable intermediate representation of a straight-line program over
  the ``Interval`` primitives ``literal`` / ``sub`` / ``mul``. This is
  **not** a speculative universal IR: it covers exactly the vocabulary the
  one instrumented call site in this module (:func:`record_ldlt_diagonal_trace`)
  needs. Adding ``add`` / ``div`` / ``sqrt`` / ``reciprocal`` replay support
  is future work, gated on a concrete call site and a matching
  ``ZInterval`` kernel lemma -- see ``Omnibias/Replay.lean``.
* :class:`DomainSubdivisionCertificate` -- the finite list of leaf boxes a
  branch-and-bound / bisection search actually visited, plus a
  :meth:`~DomainSubdivisionCertificate.covers_no_gaps` combinatorial check
  that they tile the claimed starting domain edge-to-edge.

Recording is a pure **side channel**: :class:`ReplayRecorder` and
:func:`record_ldlt_diagonal_trace` are new, independent code paths that do
not import, call, wrap, or otherwise alter
:mod:`omnibias.core.verified.eig_operator`. No existing function's
signature, default behavior, or return value changes because this module
exists; a caller who never constructs a :class:`ReplayRecorder` is
unaffected.

``omnibias.core.proof.lean_check`` extends its existing obligation
dispatch to recognize the ``interval_replay_trace`` and
``domain_subdivision`` payload shapes emitted here (:meth:`ReplayTrace.to_payload`,
:meth:`DomainSubdivisionCertificate.to_payload`) and emits a Lean obligation
against the generic, reusable checkers in ``Omnibias/Replay.lean`` /
``Omnibias/Subdivision.lean``. ``theorem_prover_verified`` is earned only by
that genuine ``lake build``; this module never sets it itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from omnibias.core.proof.certificate import make_certificate
from omnibias.core.proof.lean_check import LeanCheckResult, check_certificate
from omnibias.core.verified.interval import Interval, IntervalLike

#: The replay-trace operation vocabulary this module supports. Scoped to
#: exactly what :func:`record_ldlt_diagonal_trace` needs -- see the module
#: docstring for why this is intentionally not a bigger universal IR.
ReplayOp = Literal["literal", "sub", "mul"]

_REPLAY_OPS: frozenset[str] = frozenset({"literal", "sub", "mul"})

#: A symmetric matrix of interval-like entries, as accepted by
#: :func:`record_ldlt_diagonal_trace` (mirrors ``eig_operator.Matrix``).
Matrix = Sequence[Sequence[IntervalLike]]


@dataclass(frozen=True)
class ReplayStep:
    """One step of a straight-line ``Interval`` derivation.

    ``op`` is ``"literal"`` (a trusted input, not derived from any prior
    step -- ``args`` is empty), ``"sub"``, or ``"mul"`` (each with exactly
    two prior-step indices in ``args``). ``lo``/``hi`` are the *actual*
    ``Interval`` endpoints Python computed for this step: this dataclass
    only records what already happened, it never recomputes or re-derives
    anything itself.
    """

    op: ReplayOp
    args: tuple[int, ...]
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.op not in _REPLAY_OPS:
            raise ValueError(f"unsupported replay op {self.op!r}; expected one of {sorted(_REPLAY_OPS)}")
        want = 0 if self.op == "literal" else 2
        if len(self.args) != want:
            raise ValueError(f"{self.op!r} step expects {want} operand indices, got {len(self.args)}")
        if self.lo > self.hi:
            raise ValueError(f"empty recorded interval: lo={self.lo!r} > hi={self.hi!r}")

    def value(self) -> Interval:
        return Interval(self.lo, self.hi)

    def to_json(self) -> dict[str, object]:
        """A JSON-safe dict using the repository's ``float.hex()`` convention
        for exact endpoint serialization (matching
        :mod:`omnibias.core.proof.certificate`)."""
        return {
            "op": self.op,
            "args": list(self.args),
            "lo": float(self.lo).hex(),
            "hi": float(self.hi).hex(),
        }


@dataclass(frozen=True)
class ReplayTrace:
    """A finite, ordered straight-line trace over ``Interval`` primitives.

    ``steps[i].args`` may only index strictly earlier entries (``< i``), so
    the trace is a genuine straight-line program, safe to replay
    front-to-back with no forward references. ``conclusions`` names the
    subset of step indices that are the run's sound "answers" (e.g. each
    pivot ``D_jj``) -- bookkeeping for callers and for the sealed
    certificate payload; it is not interpreted by the replay checker
    itself, which re-derives *every* ``sub``/``mul`` step regardless of
    whether it is a named conclusion.

    Validation here is purely **structural** (well-formed shape, in-range
    operand indices): it does *not* check that any step's recorded
    envelope is the arithmetically correct composition of its operands.
    That is exactly what the emitted Lean obligation
    (``Omnibias.Replay.replayOk ... := by decide``) checks, genuinely, in
    the Lean kernel -- deliberately independent of this Python
    implementation.
    """

    steps: tuple[ReplayStep, ...]
    conclusions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("ReplayTrace requires at least one step")
        n = len(self.steps)
        for i, step in enumerate(self.steps):
            for a in step.args:
                if not (0 <= a < i):
                    raise ValueError(f"step {i} ({step.op!r}) references non-prior index {a}")
        for c in self.conclusions:
            if not (0 <= c < n):
                raise ValueError(f"conclusion index {c} out of range [0, {n})")

    def value(self, index: int) -> Interval:
        return self.steps[index].value()

    def conclusion_values(self) -> tuple[Interval, ...]:
        return tuple(self.value(i) for i in self.conclusions)

    def to_payload(self) -> dict[str, object]:
        """The ``interval_replay_trace`` certificate payload
        :mod:`omnibias.core.proof.lean_check` dispatches on."""
        return {
            "type": "interval_replay_trace",
            "steps": [s.to_json() for s in self.steps],
            "conclusions": list(self.conclusions),
        }


class ReplayRecorder:
    """Append-only, opt-in recorder of ``Interval`` primitive operations.

    Instantiate one per run, feed it operands the caller already has (as
    prior step indices from :meth:`literal` / :meth:`sub` / :meth:`mul`),
    and call :meth:`trace` at the end. The recorder does not intercept,
    wrap, or monkeypatch any existing function -- it is a plain object a
    caller opts into using; a caller that never constructs one changes no
    behavior for anyone else.

    Every arithmetic method calls the real ``Interval`` operator (never a
    reimplementation), so the recorded endpoints are bit-for-bit identical
    to what an uninstrumented computation would have produced.
    """

    def __init__(self) -> None:
        self._steps: list[ReplayStep] = []

    def __len__(self) -> int:
        return len(self._steps)

    def value(self, index: int) -> Interval:
        return self._steps[index].value()

    def literal(self, value: IntervalLike) -> int:
        """Record a trusted input (a matrix entry, or a value derived by an
        operation outside this trace's vocabulary, e.g. ``reciprocal``)."""
        iv = Interval.from_value(value)
        self._steps.append(ReplayStep("literal", (), iv.lo, iv.hi))
        return len(self._steps) - 1

    def sub(self, a: int, b: int) -> int:
        result = self.value(a) - self.value(b)
        self._steps.append(ReplayStep("sub", (a, b), result.lo, result.hi))
        return len(self._steps) - 1

    def mul(self, a: int, b: int) -> int:
        result = self.value(a) * self.value(b)
        self._steps.append(ReplayStep("mul", (a, b), result.lo, result.hi))
        return len(self._steps) - 1

    def trace(self, conclusions: Sequence[int] = ()) -> ReplayTrace:
        return ReplayTrace(tuple(self._steps), tuple(conclusions))


def record_ldlt_diagonal_trace(matrix: Matrix) -> tuple[tuple[Interval, ...] | None, ReplayTrace]:
    r"""Instrumented mirror of the pivot recurrence in
    ``omnibias.core.verified.eig_operator._ldlt_pivots``, recording a
    :class:`ReplayTrace` of every ``sub``/``mul`` step alongside each
    trusted literal input.

    This is a **thin, independent copy** of that recurrence, not a wrapper
    around it: it does not import, call, or alter
    :mod:`omnibias.core.verified.eig_operator` in any way, so adding it can
    never change that module's behavior for any existing caller. It is
    verified (in ``tests/proof/test_replay.py``) to reproduce
    ``interval_ldlt_pivots(matrix)`` bit-for-bit on the same input.

    Trust boundary (mirrors the one ``Omnibias/LDLT.lean`` already
    documents for the division-bearing factorisation): the reciprocal
    ``1 / D_jj`` needed for each off-diagonal ``L`` entry is *not* itself
    replayed -- ``ZInterval`` has no division -- so it is recorded as a
    trusted :meth:`~ReplayRecorder.literal`, exactly like a raw matrix
    entry. Only the ``sub``/``mul`` steps of the diagonal-pivot and
    off-diagonal-factor recurrences are genuinely replayed by the Lean
    checker.

    Returns ``(None, trace)`` as soon as a pivot interval straddles zero
    (mirroring ``_ldlt_pivots``'s own ``None`` early return -- the trace
    still records every step taken before that point), or ``(pivots,
    trace)`` with the full length-``n`` pivot tuple.
    """
    rows = [[Interval.from_value(x) for x in row] for row in matrix]
    n = len(rows)
    if n == 0:
        raise ValueError("matrix must be non-empty")
    for row in rows:
        if len(row) != n:
            raise ValueError("matrix must be square")

    rec = ReplayRecorder()
    lmat: list[list[Interval]] = [[Interval.point(0.0) for _ in range(n)] for _ in range(n)]
    d: list[Interval] = [Interval.point(0.0) for _ in range(n)]
    l_index: list[list[int]] = [[-1] * n for _ in range(n)]
    d_index: list[int] = [-1] * n
    conclusions: list[int] = []

    for j in range(n):
        lmat[j][j] = Interval.point(1.0)
        dj = rows[j][j]
        dj_idx = rec.literal(dj)
        for k in range(j):
            sq_idx = rec.mul(l_index[j][k], l_index[j][k])
            term_idx = rec.mul(sq_idx, d_index[k])
            dj_idx = rec.sub(dj_idx, term_idx)
            dj = dj - lmat[j][k] * lmat[j][k] * d[k]
        if dj.lo <= 0.0 <= dj.hi:
            return None, rec.trace(tuple(conclusions))
        d[j] = dj
        d_index[j] = dj_idx
        conclusions.append(dj_idx)
        inv = dj.reciprocal()
        inv_idx = rec.literal(inv)
        for i in range(j + 1, n):
            lij = rows[i][j]
            lij_idx = rec.literal(lij)
            for k in range(j):
                sq_idx = rec.mul(l_index[i][k], l_index[j][k])
                term_idx = rec.mul(sq_idx, d_index[k])
                lij_idx = rec.sub(lij_idx, term_idx)
                lij = lij - lmat[i][k] * lmat[j][k] * d[k]
            final_idx = rec.mul(lij_idx, inv_idx)
            lmat[i][j] = lij * inv
            l_index[i][j] = final_idx

    return tuple(d), rec.trace(tuple(conclusions))


@dataclass(frozen=True)
class SubdivisionLeaf:
    """One visited leaf of a branch-and-bound / bisection search.

    ``lo``/``hi`` are integer sub-range boundaries of ``[0, resolution]``
    (the box ``lo / resolution .. hi / resolution`` of the claimed
    starting domain, over the shared ``resolution`` of the enclosing
    :class:`DomainSubdivisionCertificate`). ``conclusion`` is a short,
    human-readable label for whatever sound, per-leaf fact the search
    established there (e.g. ``"eigenvalue count below midpoint >= k"``);
    it is not interpreted by the coverage check, which only verifies the
    leaves tile the domain with no gap.
    """

    lo: int
    hi: int
    conclusion: str = ""

    def __post_init__(self) -> None:
        if self.lo >= self.hi:
            raise ValueError(f"leaf must be non-empty: lo={self.lo} >= hi={self.hi}")


@dataclass(frozen=True)
class DomainSubdivisionCertificate:
    """Finite coverage certificate for a branch-and-bound leaf list.

    ``resolution`` fixes the common integer denominator for
    ``[0, resolution]``; ``leaves`` is the list of visited leaves **in the
    order they must tile the domain** (sorted by ``lo``; the caller is
    responsible for that ordering, since visitation order in a real
    branch-and-bound search need not match left-to-right order).

    :meth:`covers_no_gaps` exactly mirrors the combinatorics of
    ``Omnibias.Subdivision.coversNoGaps`` in
    ``formal/omnibias-verified-kernel``, so Python and Lean agree by
    construction, not by coincidence -- but it is a convenience self-check
    only. The authoritative gate is the emitted Lean obligation
    (``coversNoGaps ... := by decide``), genuinely re-derived by the Lean
    kernel; :mod:`omnibias.core.proof.lean_check` additionally refuses to
    emit that obligation at all when this Python-side check already fails,
    exactly like the existing positive-definite-pivot and poisedness
    obligations gate on a Python-side truth check before asking Lean to
    re-derive it.
    """

    resolution: int
    leaves: tuple[SubdivisionLeaf, ...]

    def __post_init__(self) -> None:
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if not self.leaves:
            raise ValueError("DomainSubdivisionCertificate requires at least one leaf")

    def covers_no_gaps(self) -> bool:
        """``True`` iff the leaves tile ``[0, resolution]`` edge-to-edge:
        the first leaf starts at ``0``, each leaf starts exactly where the
        previous one ended, and the last leaf ends at ``resolution``."""
        first = self.leaves[0]
        if first.lo != 0:
            return False
        prev_hi = first.lo
        for leaf in self.leaves:
            if leaf.lo != prev_hi:
                return False
            prev_hi = leaf.hi
        return prev_hi == self.resolution

    def to_payload(self) -> dict[str, object]:
        """The ``domain_subdivision`` certificate payload
        :mod:`omnibias.core.proof.lean_check` dispatches on."""
        return {
            "type": "domain_subdivision",
            "resolution": self.resolution,
            "leaves": [
                {"lo": leaf.lo, "hi": leaf.hi, "conclusion": leaf.conclusion} for leaf in self.leaves
            ],
        }


@dataclass(frozen=True)
class ReplayCertificateReport:
    """A sealed certificate plus the Lean-kernel verdict for one replay-style
    obligation (an :class:`ReplayTrace` or a :class:`DomainSubdivisionCertificate`).

    ``theorem_prover_verified`` reuses the exact flag every other omnibias
    certificate kind earns from :func:`omnibias.core.proof.lean_check.check_certificate`
    -- it means the same thing here as everywhere else: a genuine ``lake
    build`` re-checked this certificate's finite obligation and accepted
    it. It is ``False`` whenever the Lean toolchain is unavailable or the
    kernel rejected the obligation; this module never sets it itself.
    """

    certificate: dict[str, Any]
    theorem_prover_verified: bool
    lean: LeanCheckResult | None


def seal_replay_certificate(
    trace: ReplayTrace, *, claim: str, run_lean: bool = True
) -> ReplayCertificateReport:
    """Seal ``trace`` as a v1 certificate and, if requested, drive the Lean
    replay checker (``Omnibias.Replay.replayOk``).

    ``run_lean=False`` seals the certificate without invoking ``lake`` at
    all (useful for structural-only tests); ``run_lean=True`` (the
    default) calls :func:`omnibias.core.proof.lean_check.check_certificate`,
    which itself degrades gracefully -- ``available=False``,
    ``verified=False`` -- when no Lean toolchain is present, exactly like
    every other obligation kind.
    """
    cert = make_certificate(claim=claim, payload=trace.to_payload())
    lean: LeanCheckResult | None = None
    verified = False
    if run_lean:
        lean = check_certificate(cert)
        verified = bool(lean.verified)
    return ReplayCertificateReport(certificate=cert, theorem_prover_verified=verified, lean=lean)


def seal_domain_subdivision_certificate(
    subdivision: DomainSubdivisionCertificate, *, claim: str, run_lean: bool = True
) -> ReplayCertificateReport:
    """Seal ``subdivision`` as a v1 certificate and, if requested, drive the
    Lean coverage checker (``Omnibias.Subdivision.coversNoGaps``).

    Same graceful-degradation contract as :func:`seal_replay_certificate`.
    """
    cert = make_certificate(claim=claim, payload=subdivision.to_payload())
    lean: LeanCheckResult | None = None
    verified = False
    if run_lean:
        lean = check_certificate(cert)
        verified = bool(lean.verified)
    return ReplayCertificateReport(certificate=cert, theorem_prover_verified=verified, lean=lean)


__all__ = [
    "DomainSubdivisionCertificate",
    "Matrix",
    "ReplayCertificateReport",
    "ReplayOp",
    "ReplayRecorder",
    "ReplayStep",
    "ReplayTrace",
    "SubdivisionLeaf",
    "record_ldlt_diagonal_trace",
    "seal_domain_subdivision_certificate",
    "seal_replay_certificate",
]
