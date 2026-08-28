# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""phase1-lean-replay: finite straight-line replay traces + domain-subdivision
coverage certificates, and their Lean-kernel bridge.

Runs with or without a Lean toolchain: structural validation, obligation
generation, and the Python-side gap refusal are always exercised; the actual
``lake build`` pass (and ``theorem_prover_verified=True``) is asserted only
when ``lake`` is present -- following the exact pattern already used by
``test_lean_check.py`` / ``test_rational_stencil.py``.
"""

from __future__ import annotations

import pytest
from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.core.proof.lean_check import (
    check_certificate,
    generate_obligation,
    lean_check_available,
)
from omnibias.core.proof.replay import (
    DomainSubdivisionCertificate,
    ReplayStep,
    ReplayTrace,
    SubdivisionLeaf,
    record_ldlt_diagonal_trace,
    seal_domain_subdivision_certificate,
    seal_replay_certificate,
)
from omnibias.core.verified.eig_operator import interval_ldlt_pivots

# --------------------------------------------------------------------------- #
# ReplayTrace / ReplayRecorder: pure Python, no Lean involved.
# --------------------------------------------------------------------------- #
_SPD_3X3 = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]]
_SPD_2X2 = [[4.0, 1.0], [1.0, 3.0]]
_SINGULAR_2X2 = [[1.0, 1.0], [1.0, 1.0]]


def test_record_ldlt_diagonal_trace_matches_direct_computation_bit_for_bit() -> None:
    """Recording is a true no-op on the underlying arithmetic: the traced
    pivots are bit-for-bit identical to the uninstrumented
    ``interval_ldlt_pivots`` on the same input."""
    direct = interval_ldlt_pivots(_SPD_3X3)
    traced, trace = record_ldlt_diagonal_trace(_SPD_3X3)
    assert direct is not None and traced is not None
    assert len(direct) == len(traced) == 3
    for expected, got in zip(direct, traced, strict=True):
        assert expected.lo == got.lo
        assert expected.hi == got.hi
    # Every named conclusion's value matches the corresponding pivot.
    assert trace.conclusion_values() == traced


def test_record_ldlt_diagonal_trace_is_independent_of_eig_operator() -> None:
    """Calling the instrumented recorder never touches, imports as a side
    effect, or otherwise perturbs ``eig_operator`` -- an uninstrumented call
    before and after recording returns the identical result."""
    before = interval_ldlt_pivots(_SPD_2X2)
    record_ldlt_diagonal_trace(_SPD_2X2)
    after = interval_ldlt_pivots(_SPD_2X2)
    assert before == after


def test_record_ldlt_diagonal_trace_none_on_straddling_pivot() -> None:
    """A singular matrix box drives a pivot to straddle zero; both the
    uninstrumented function and the recorder report ``None``, and the trace
    still records every step taken before the early return."""
    assert interval_ldlt_pivots(_SINGULAR_2X2) is None
    pivots, trace = record_ldlt_diagonal_trace(_SINGULAR_2X2)
    assert pivots is None
    assert len(trace.steps) > 0
    assert len(trace.conclusions) == 1  # only d[0] was certified before the straddle


def test_replay_trace_structural_validation() -> None:
    with pytest.raises(ValueError, match="non-prior index"):
        ReplayTrace((ReplayStep("sub", (0, 1), 0.0, 1.0),))
    with pytest.raises(ValueError, match="expects 0 operand"):
        ReplayStep("literal", (1,), 0.0, 1.0)
    with pytest.raises(ValueError, match="expects 2 operand"):
        ReplayStep("mul", (0,), 0.0, 1.0)
    with pytest.raises(ValueError, match="unsupported replay op"):
        ReplayStep("div", (), 0.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ReplayTrace(())  # at least one step required
    with pytest.raises(ValueError, match="out of range"):
        ReplayTrace((ReplayStep("literal", (), 0.0, 1.0),), conclusions=(5,))


def test_replay_trace_to_payload_shape() -> None:
    _, trace = record_ldlt_diagonal_trace(_SPD_2X2)
    payload = trace.to_payload()
    assert payload["type"] == "interval_replay_trace"
    steps = payload["steps"]
    assert isinstance(steps, list) and len(steps) == len(trace.steps)
    for step in steps:
        assert set(step) == {"op", "args", "lo", "hi"}
        assert isinstance(step["lo"], str) and isinstance(step["hi"], str)
        # round-trips exactly through float.hex()
        float.fromhex(step["lo"])
        float.fromhex(step["hi"])


# --------------------------------------------------------------------------- #
# DomainSubdivisionCertificate: pure Python coverage combinatorics.
# --------------------------------------------------------------------------- #
def test_domain_subdivision_covers_no_gaps_true_for_a_clean_partition() -> None:
    leaves = (
        SubdivisionLeaf(0, 4, "count_below(1/4) == 0"),
        SubdivisionLeaf(4, 10, "count_below(1/2) == 1"),
        SubdivisionLeaf(10, 16, "count_below(1) == 2"),
    )
    cert = DomainSubdivisionCertificate(resolution=16, leaves=leaves)
    assert cert.covers_no_gaps() is True


@pytest.mark.parametrize(
    "leaves,resolution",
    [
        # a gap between the first and second leaf
        ((SubdivisionLeaf(0, 4, "a"), SubdivisionLeaf(5, 10, "b")), 10),
        # first leaf does not start at 0
        ((SubdivisionLeaf(1, 4, "a"), SubdivisionLeaf(4, 10, "b")), 10),
        # last leaf does not end at resolution
        ((SubdivisionLeaf(0, 4, "a"), SubdivisionLeaf(4, 8, "b")), 10),
        # overlapping leaves
        ((SubdivisionLeaf(0, 5, "a"), SubdivisionLeaf(4, 10, "b")), 10),
    ],
)
def test_domain_subdivision_covers_no_gaps_false_cases(
    leaves: tuple[SubdivisionLeaf, ...], resolution: int
) -> None:
    cert = DomainSubdivisionCertificate(resolution=resolution, leaves=leaves)
    assert cert.covers_no_gaps() is False


def test_domain_subdivision_structural_validation() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SubdivisionLeaf(5, 3)
    with pytest.raises(ValueError, match="resolution must be positive"):
        DomainSubdivisionCertificate(resolution=0, leaves=(SubdivisionLeaf(0, 1),))
    with pytest.raises(ValueError, match="at least one leaf"):
        DomainSubdivisionCertificate(resolution=10, leaves=())


# --------------------------------------------------------------------------- #
# Lean-kernel bridge: obligation generation (always exercised).
# --------------------------------------------------------------------------- #
def test_generate_replay_obligation_contains_expected_lean() -> None:
    _, trace = record_ldlt_diagonal_trace(_SPD_2X2)
    cert = make_certificate(claim="LDLT diagonal-pivot replay", payload=trace.to_payload())
    src = generate_obligation(cert)
    assert src is not None
    assert "import Omnibias.Replay" in src
    assert "Omnibias.Replay.replayOk" in src
    assert "by decide" in src
    assert src.count("⟨") == len(trace.steps)


def test_generate_domain_subdivision_obligation_contains_expected_lean() -> None:
    leaves = (SubdivisionLeaf(0, 4, "a"), SubdivisionLeaf(4, 10, "b"))
    subdivision = DomainSubdivisionCertificate(resolution=10, leaves=leaves)
    cert = make_certificate(claim="bisection coverage", payload=subdivision.to_payload())
    src = generate_obligation(cert)
    assert src is not None
    assert "import Omnibias.Subdivision" in src
    assert "Omnibias.Subdivision.coversNoGaps" in src
    assert "by decide" in src


def test_generate_domain_subdivision_obligation_none_for_a_gap() -> None:
    """The gapped negative control is caught at the **Python export layer**:
    ``generate_obligation`` refuses to emit Lean source at all, so no
    ``lake build`` is ever invoked for a certificate that already fails the
    combinatorial coverage check."""
    leaves = (SubdivisionLeaf(0, 4, "a"), SubdivisionLeaf(5, 10, "b"))  # gap: 4..5
    gapped = DomainSubdivisionCertificate(resolution=10, leaves=leaves)
    assert gapped.covers_no_gaps() is False
    cert = make_certificate(claim="gapped bisection", payload=gapped.to_payload())
    assert generate_obligation(cert) is None
    result = check_certificate(cert)
    assert result.verified is False
    assert result.obligation == ""
    assert "no finite Lean-checkable obligation" in result.detail


def test_generate_replay_obligation_none_for_malformed_payload() -> None:
    bad_shape = make_certificate(
        claim="x", payload={"type": "interval_replay_trace", "steps": []}
    )
    assert generate_obligation(bad_shape) is None
    bad_index = make_certificate(
        claim="x",
        payload={
            "type": "interval_replay_trace",
            "steps": [{"op": "sub", "args": [0, 1], "lo": (0.0).hex(), "hi": (1.0).hex()}],
        },
    )
    assert generate_obligation(bad_index) is None


# --------------------------------------------------------------------------- #
# Full pipeline: seal + (when available) genuine `lake build`.
# --------------------------------------------------------------------------- #
def test_replay_certificate_seal_degrades_gracefully() -> None:
    _, trace = record_ldlt_diagonal_trace(_SPD_3X3)
    report = seal_replay_certificate(
        trace, claim="LDLT diagonal-pivot recurrence replay for a 3x3 SPD matrix box"
    )
    assert verify_certificate_digest(report.certificate)
    assert "theorem_prover_verified" not in report.certificate.get("honesty", {})
    if lean_check_available():  # pragma: no cover - Lean-equipped environment only
        assert report.theorem_prover_verified is True
        assert report.lean is not None and report.lean.verified is True
    else:
        assert report.theorem_prover_verified is False
        assert report.lean is not None
        assert report.lean.available is False


def test_domain_subdivision_certificate_seal_degrades_gracefully() -> None:
    leaves = (
        SubdivisionLeaf(0, 4, "count_below(1/4) == 0"),
        SubdivisionLeaf(4, 10, "count_below(1/2) == 1"),
        SubdivisionLeaf(10, 16, "count_below(1) == 2"),
    )
    cert_obj = DomainSubdivisionCertificate(resolution=16, leaves=leaves)
    report = seal_domain_subdivision_certificate(
        cert_obj, claim="bisection leaves cover [0, 16] with no gap"
    )
    if lean_check_available():  # pragma: no cover - Lean-equipped environment only
        assert report.theorem_prover_verified is True
        assert report.lean is not None and report.lean.verified is True
    else:
        assert report.theorem_prover_verified is False
        assert report.lean is not None
        assert report.lean.available is False


def test_replay_trace_tampered_conclusion_is_rejected_by_lean_kernel() -> None:
    """The negative control for **semantic** tampering: a recorded ``sub``
    step's endpoint is shifted to a value that no longer contains the true
    recomputation. ``ReplayTrace``/``generate_obligation`` have no way to
    catch this structurally (the shape is still perfectly well-formed) --
    only ``lake build``'s ``decide`` genuinely re-derives the arithmetic and
    rejects it. This is the intentional design point: the Lean pass is a
    real, independent check, not a rubber stamp of whatever Python recorded.
    """
    _, trace = record_ldlt_diagonal_trace(_SPD_2X2)
    payload = trace.to_payload()
    target = trace.conclusions[-1]
    assert trace.steps[target].op != "literal"  # a genuinely re-derived step
    step = dict(payload["steps"][target])
    true_lo, true_hi = float.fromhex(step["lo"]), float.fromhex(step["hi"])
    step["lo"] = float(true_lo + 1000.0).hex()
    step["hi"] = float(true_hi + 1000.0).hex()
    payload["steps"] = [*payload["steps"][:target], step, *payload["steps"][target + 1 :]]
    cert = make_certificate(claim="tampered replay trace", payload=payload)
    src = generate_obligation(cert)
    assert src is not None  # the obligation is still emitted -- Lean is the arbiter
    result = check_certificate(cert)
    if lean_check_available():  # pragma: no cover - Lean-equipped environment only
        assert result.verified is False
        assert result.available is True
    else:
        assert result.available is False
        assert result.verified is False
