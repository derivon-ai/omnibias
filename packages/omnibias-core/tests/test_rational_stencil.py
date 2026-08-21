# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 01-11: rational stencil obligations and the Lean-kernel bridge."""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.multipack import MultiPackSpec, PackSpec
from omnibias.core.proof import (
    generate_obligation,
    kernel_root,
    lean_check_available,
    poisedness_obligation,
    seal_poisedness_certificate,
    seal_stencil_certificate,
    stencil_consistency_obligation,
)
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.proof.lean_check import check_certificate
from omnibias.core.proof.obligations.rational_stencil import (
    PAYLOAD_POISEDNESS,
    PAYLOAD_STENCIL,
    RationalStencil,
    birkhoff_spec_multipack,
    birkhoff_spec_stencil,
    curated_rational_stencils,
    honesty_payload,
)


def test_curated_set_is_consistent_and_large() -> None:
    stencils = curated_rational_stencils()
    assert len(stencils) >= 20
    names = [s.name for s in stencils]
    assert len(names) == len(set(names))
    for st in stencils:
        obl = stencil_consistency_obligation(st)
        assert obl.holds, st.name
        assert obl.kind == PAYLOAD_STENCIL
        n = st.n_conditions
        for j in range(n):
            rhs = Fraction(1) if j == st.target_order else Fraction(0)
            assert st.moment_sum(j) == rhs, (st.name, j)


def test_birkhoff_spec_moments() -> None:
    st = birkhoff_spec_stencil()
    assert st.moment_sum(0) == 0
    assert st.moment_sum(1) == 1
    assert st.moment_sum(2) == 0
    assert st.computed_leading() == Fraction(5, 18)
    assert stencil_consistency_obligation(st).holds


def test_birkhoff_poisedness_det() -> None:
    obl = poisedness_obligation(birkhoff_spec_stencil())
    assert obl.holds
    assert obl.payload["det"] == ["3", "2"]
    via_mp = poisedness_obligation(birkhoff_spec_multipack())
    assert via_mp.holds
    assert via_mp.payload["det"] == ["3", "2"]


def test_polya_failure_is_not_poised() -> None:
    spec = MultiPackSpec.from_packs((PackSpec(order=2, mean=0.0),))
    obl = poisedness_obligation(spec)
    assert obl.holds is False
    assert obl.payload["type"] == PAYLOAD_POISEDNESS


def test_corrupted_weights_fail_python_algebra() -> None:
    bad = birkhoff_spec_stencil().corrupt_first_weight()
    obl = stencil_consistency_obligation(bad)
    assert obl.holds is False
    src = generate_obligation(
        seal_stencil_certificate(bad, run_lean=False).certificate
    )
    assert src is not None
    assert "allRatEq" in src


def test_generate_stencil_obligation_shape() -> None:
    cert = seal_stencil_certificate(birkhoff_spec_stencil(), run_lean=False).certificate
    src = generate_obligation(cert)
    assert src is not None
    assert "import Omnibias.RationalStencil" in src
    assert "allRatEq" in src
    assert "theorem obligation" in src
    assert "by decide" in src
    lowered = src.lower()
    assert "limit" not in lowered
    assert "sorry" not in lowered
    assert "mathlib" not in lowered


def test_generate_poisedness_obligation_shape() -> None:
    cert = seal_poisedness_certificate(birkhoff_spec_stencil(), run_lean=False).certificate
    src = generate_obligation(cert)
    assert src is not None
    assert "allIntGe" in src
    assert "ratNez" in src
    assert "limit" not in src.lower()


def test_g3_no_toolchain_degrades() -> None:
    report = seal_stencil_certificate(birkhoff_spec_stencil(), run_lean=True)
    assert verify_certificate_digest(report.certificate)
    assert report.mathlib_verified is False
    assert "theorem_prover_verified" not in report.certificate.get("honesty", {})
    if lean_check_available():
        assert report.theorem_prover_verified is True
        assert report.lean is not None and report.lean.verified is True
    else:
        assert report.theorem_prover_verified is False
        assert report.lean is not None and report.lean.verified is False
        assert report.lean.available is False


def test_g4_tamper_evidence() -> None:
    report = seal_stencil_certificate(birkhoff_spec_stencil(), run_lean=False)
    cert = dict(report.certificate)
    payload = dict(cert["payload"])
    payload["target_order"] = 99
    cert["payload"] = payload
    assert verify_certificate_digest(report.certificate)
    assert verify_certificate_digest(cert) is False
    result = check_certificate(cert)
    assert result.verified is False
    assert result.obligation == ""
    assert "digest" in result.detail


def test_g5_tier_separation() -> None:
    st_report = seal_stencil_certificate(birkhoff_spec_stencil(), run_lean=True)
    po_report = seal_poisedness_certificate(birkhoff_spec_stencil(), run_lean=True)
    assert st_report.mathlib_verified is False
    assert po_report.mathlib_verified is False
    honesty = honesty_payload()
    assert honesty["mathlib_path_used"] is False
    assert honesty["collapse_limit_in_lean"] is False


def test_g1_g2_kernel_when_available() -> None:
    stencils = curated_rational_stencils()
    assert len(stencils) >= 20
    if not lean_check_available():
        for st in stencils:
            src = generate_obligation(
                seal_stencil_certificate(st, run_lean=False).certificate
            )
            assert src is not None
        return
    for st in stencils:
        report = seal_stencil_certificate(st, run_lean=True)
        assert report.theorem_prover_verified is True, st.name
        assert report.mathlib_verified is False
    bad = seal_stencil_certificate(
        birkhoff_spec_stencil().corrupt_first_weight(), run_lean=True
    )
    assert bad.obligation.holds is False
    assert bad.theorem_prover_verified is False
    assert bad.lean is not None and bad.lean.verified is False
    poised = seal_poisedness_certificate(birkhoff_spec_stencil(), run_lean=True)
    assert poised.theorem_prover_verified is True
    assert poised.mathlib_verified is False


def test_lean_sources_have_no_sorry_or_limit() -> None:
    root = kernel_root()
    assert root is not None
    text = (root / "Omnibias" / "RationalStencil.lean").read_text(encoding="utf-8")
    assert "sorry" not in text
    assert "limit" not in text.lower()


def test_cannot_forge_theorem_prover_verified_into_seal() -> None:
    from omnibias.core.proof.certificate import make_certificate

    try:
        make_certificate(
            claim="forged",
            payload={"type": PAYLOAD_STENCIL, "conditions": []},
            honesty={"theorem_prover_verified": True},
        )
    except ValueError as exc:
        assert "theorem_prover_verified" in str(exc)
    else:
        raise AssertionError("forged formal flag must be refused")


def test_malformed_stencil_payload_emits_nothing() -> None:
    from omnibias.core.proof.certificate import make_certificate

    cert = make_certificate(
        claim="x",
        payload={"type": PAYLOAD_STENCIL, "conditions": []},
        honesty={},
    )
    assert generate_obligation(cert) is None
