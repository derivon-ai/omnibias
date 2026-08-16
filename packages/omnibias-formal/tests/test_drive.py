# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for the deterministic formal-loop driver (``omnibias.formal.drive``).

These run with or without a Lean toolchain: classification, failure distillation,
the not-attempted / graceful-degradation paths, and the honesty invariant (the
``mathlib_verified`` tier is never set without a genuine ``lake`` pass) are always
exercised; the actual ``verified=True`` path is asserted only when ``lake`` + the
analytic checkout are present (the dedicated ``lean-analytic`` CI job).
"""

from __future__ import annotations

from typing import Any

from omnibias.core.proof.certificate import (
    interval_certificate,
    make_certificate,
    positive_definite_certificate,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import radii_polynomial_certificate
from omnibias.formal import (
    MATHLIB_CLAIM_KEY,
    DriveReport,
    casimir_certificate,
    classify_obligation,
    compact_box_certificate,
    drive_obligation,
    enclosure_trace_certificate,
    generate_obligation,
    haar_certificate,
    mathlib_check_available,
    named_zero_certificate,
    nk_existence_certificate,
    polymer_certificate,
    sixj_certificate,
    tower_coeffs_certificate,
)
from omnibias.formal.drive import _summarize_lake_failure


def _krawczyk_cert(kappa: float = 0.25) -> dict[str, Any]:
    return make_certificate(
        claim="unique zero of F in the Krawczyk box",
        payload={
            "type": "krawczyk",
            "radius": 0.5,
            "kappa": kappa,
            "center": [1.0, 2.0],
            "enclosure": [[0.9, 1.1], [1.8, 2.2]],
        },
    )


# --------------------------------------------------------------------------- #
# classify_obligation: cheap, Lean-free triage.
# --------------------------------------------------------------------------- #
def test_classify_interval_sign() -> None:
    assert classify_obligation(interval_certificate("q", Interval(0.5, 2.0))) == "sign"


def test_classify_positive_definite() -> None:
    cert = positive_definite_certificate("pd", [Interval(1.0, 1.5), Interval(0.5, 0.75)])
    assert classify_obligation(cert) == "positive_definite"


def test_classify_radii_polynomial() -> None:
    rc = radii_polynomial_certificate(0.001, 0.05, 0.0, 0.05)
    assert rc is not None
    assert classify_obligation(rc.certificate) == "radii_polynomial"


def test_classify_krawczyk() -> None:
    assert classify_obligation(_krawczyk_cert()) == "krawczyk"


def test_classify_tower_coeffs() -> None:
    assert classify_obligation(tower_coeffs_certificate("sigmoid", 2)) == "tower_coeffs"


def test_classify_nk_existence() -> None:
    assert classify_obligation(nk_existence_certificate("radii")) == "nk_existence"


def test_classify_enclosure_trace() -> None:
    assert classify_obligation(enclosure_trace_certificate("nk")) == "enclosure_trace"


def test_classify_named_zero() -> None:
    assert classify_obligation(named_zero_certificate("circle_line")) == "named_zero"


def test_classify_compact_box() -> None:
    assert classify_obligation(compact_box_certificate("ns_box")) == "compact_box"


def test_classify_casimir() -> None:
    assert classify_obligation(casimir_certificate("su2_fund")) == "casimir"


def test_classify_polymer() -> None:
    assert classify_obligation(polymer_certificate("backtrack_4")) == "polymer"


def test_classify_sixj() -> None:
    assert classify_obligation(sixj_certificate("half_half_zero")) == "sixj"


def test_classify_haar() -> None:
    assert classify_obligation(haar_certificate("weyl_prefactor_24")) == "haar_volume"


def test_classify_none_for_straddling_interval() -> None:
    assert classify_obligation(interval_certificate("q", Interval(-1.0, 1.0))) is None


def test_classify_none_for_unsupported_payload() -> None:
    assert classify_obligation({"foo": "bar"}) is None


def test_classify_agrees_with_generate_obligation() -> None:
    # classify_obligation and generate_obligation share one ordered registry, so
    # a certificate is classifiable iff an obligation is generated.
    certs: list[dict[str, Any]] = [
        interval_certificate("q", Interval(0.5, 2.0)),
        interval_certificate("q", Interval(-1.0, 1.0)),
        positive_definite_certificate("pd", [Interval(1.0, 1.5)]),
        _krawczyk_cert(),
        tower_coeffs_certificate("sigmoid", 2),
        nk_existence_certificate("radii"),
        enclosure_trace_certificate("nk"),
        named_zero_certificate("circle_line"),
        compact_box_certificate("ns_box"),
        casimir_certificate("su2_fund"),
        polymer_certificate("backtrack_4"),
        polymer_certificate("first_step_4"),
        sixj_certificate("half_half_zero"),
        haar_certificate("weyl_prefactor_24"),
        {"foo": "bar"},
    ]
    for cert in certs:
        assert (classify_obligation(cert) is None) == (generate_obligation(cert) is None)


# --------------------------------------------------------------------------- #
# _summarize_lake_failure: actionable distillation of a build log.
# --------------------------------------------------------------------------- #
_LAKE_LOG = (
    "Building OmnibiasAnalytic.Generated\n"
    "info: stdout chatter\n"
    "./OmnibiasAnalytic/Generated.lean:4:2: error: unsolved goals\n"
    "\u22a2 0 < 1\n"
    "some trailing note\n"
    "another error: type mismatch\n"
    "error: build failed\n"
)


def test_summarize_lake_failure_keeps_salient_lines() -> None:
    summary = _summarize_lake_failure(_LAKE_LOG)
    assert "unsolved goals" in summary
    assert "type mismatch" in summary
    assert "build failed" in summary
    # At most three salient lines, joined by " | " (so two separators here).
    assert summary.count(" | ") == 2
    # Non-salient chatter and the raw goal line are dropped from the summary.
    assert "stdout chatter" not in summary
    assert "0 < 1" not in summary


def test_summarize_lake_failure_falls_back_to_last_line() -> None:
    detail = "just some\noutput with no markers\nlast line here"
    assert _summarize_lake_failure(detail) == "last line here"


# --------------------------------------------------------------------------- #
# drive_obligation: one deterministic pass of the loop.
# --------------------------------------------------------------------------- #
def test_drive_interval_sign_generates_and_degrades() -> None:
    report = drive_obligation(interval_certificate("q", Interval(0.5, 2.0)))
    assert isinstance(report, DriveReport)
    assert report.obligation_class == "sign"
    assert report.attempted is True
    assert "enclosed_pos" in report.obligation
    if not mathlib_check_available():
        assert report.available is False
        assert report.verified is False
        assert report.tier is None
        assert report.failure is None
        assert "lean-analytic" in report.next_action
    else:  # pragma: no cover - only on a machine with Lean + Mathlib
        assert report.available is True
        assert report.verified is True
        assert report.tier == MATHLIB_CLAIM_KEY


def test_drive_radii_polynomial_generates() -> None:
    rc = radii_polynomial_certificate(0.001, 0.05, 0.0, 0.05)
    assert rc is not None
    report = drive_obligation(rc.certificate)
    assert report.obligation_class == "radii_polynomial"
    assert report.attempted is True
    assert "norm_num" in report.obligation
    if not mathlib_check_available():
        assert report.verified is False
        assert report.tier is None


def test_drive_unsupported_reports_not_attempted() -> None:
    report = drive_obligation({"foo": "bar"})
    assert report.obligation_class is None
    assert report.attempted is False
    assert report.verified is False
    assert report.tier is None
    assert report.obligation == ""
    assert "unsupported" in report.next_action
    assert "Mathlib-free kernel" in report.next_action


def test_drive_tampered_certificate_is_never_verified() -> None:
    # Tamper-evidence is inherited from check_certificate: a stale digest is
    # rejected before any Lean runs, so the tier can never be forged.
    cert = interval_certificate("q", Interval(0.5, 2.0))
    tampered = dict(cert)
    tampered["payload"] = {**cert["payload"], "tampered": True}
    report = drive_obligation(tampered)
    assert report.verified is False
    assert report.tier is None
    assert report.failure is not None
    assert "digest" in report.failure


def test_drive_tier_is_none_unless_verified() -> None:
    # The core honesty invariant of the driver, holds in every environment.
    report = drive_obligation(interval_certificate("q", Interval(0.5, 2.0)))
    assert (report.tier is None) == (not report.verified)
    assert report.tier in (None, MATHLIB_CLAIM_KEY)
