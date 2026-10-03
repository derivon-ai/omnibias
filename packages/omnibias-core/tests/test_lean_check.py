# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the Lean-kernel bridge (``omnibias.core.proof.lean_check``).

These run with or without a Lean toolchain: the obligation generation and the
graceful-degradation / tamper paths are always exercised; the actual ``lake``
build (and ``theorem_prover_verified=True``) is asserted only when ``lake`` is
present (e.g. the dedicated CI job).
"""

from __future__ import annotations

from omnibias.core.proof import (
    Certificate,
    Conjecture,
    FunctionProver,
    ProofAttempt,
    ProofMachine,
    check_certificate,
    generate_obligation,
    kernel_root,
    lean_check_available,
)
from omnibias.core.proof.certificate import (
    interval_certificate,
    make_certificate,
    positive_definite_certificate,
    seal_certificate,
)
from omnibias.core.verified.interval import Interval


def _rational_identity_cert(terms: list[tuple[int, int]], rhs: int) -> dict:
    """A sealed ``rational_identity`` certificate asserting ``sum_i c_i m_i = rhs``."""
    return make_certificate(
        claim="rational identity",
        payload={"type": "rational_identity", "lhs_terms": [[c, m] for c, m in terms], "rhs": rhs},
        honesty={"unproven_claim": False},
    )


def test_kernel_root_is_discoverable() -> None:
    root = kernel_root()
    assert root is not None
    assert root.name == "omnibias-verified-kernel"
    assert (root / "lakefile.lean").is_file()
    assert (root / "Omnibias" / "Certificate.lean").is_file()


def test_generate_spectral_gap_obligation() -> None:
    cert = {"subdominant_ratio_upper": 0.625, "honesty": {"unproven_claim": False}}
    src = generate_obligation(cert)
    assert src is not None
    # 0.625 == 5/8 exactly.
    assert "gapNumerator 5 8" in src
    assert "spectral_gap_pos" in src
    assert "namespace Omnibias.Generated" in src
    assert "theorem obligation" in src


def test_generate_positive_interval_obligation() -> None:
    cert = interval_certificate("q", Interval(0.5, 2.0))
    src = generate_obligation(cert)
    assert src is not None
    assert "enclosed_quantity_pos" in src
    assert "ZInterval.Mem" in src


def test_generate_negative_interval_obligation() -> None:
    cert = interval_certificate("q", Interval(-2.0, -0.5))
    src = generate_obligation(cert)
    assert src is not None
    assert "enclosed_quantity_neg" in src


def test_generate_positive_definite_obligation() -> None:
    pivots = [Interval(0.5, 0.75), Interval(2.0, 3.0), Interval(1.25, 1.5)]
    cert = positive_definite_certificate("H is PD", pivots)
    src = generate_obligation(cert)
    assert src is not None
    # The PD payload lifts the scalar shadow to the full LDL^T inertia vector.
    assert "import Omnibias.LDLT" in src
    assert "allPivotsPos" in src
    assert "by decide" in src
    assert "theorem obligation" in src
    # one pivot literal per pivot
    assert src.count("⟨") == len(pivots)


def test_generate_pd_obligation_none_when_a_pivot_is_not_positive() -> None:
    # A pivot whose lower endpoint is not strictly positive is not a PD obligation.
    straddle = positive_definite_certificate("not PD", [Interval(0.5, 1.0), Interval(-0.1, 2.0)])
    assert generate_obligation(straddle) is None
    zero_lo = positive_definite_certificate("not PD", [Interval(0.0, 1.0)])
    assert generate_obligation(zero_lo) is None


def test_pd_certificate_tamper_is_rejected_before_lean() -> None:
    cert = positive_definite_certificate("H is PD", [Interval(0.5, 0.75), Interval(2.0, 3.0)])
    tampered = dict(cert)
    tampered["payload"] = {**cert["payload"], "n": 99}  # body changed, digest stale
    result = check_certificate(tampered)
    assert result.verified is False
    assert "digest" in result.detail


def test_generate_rational_identity_obligation() -> None:
    # Bernoulli recurrence at n = 6: 1*30 + 6*(-15) + 15*5 + 20*0 + 15*(-1) + 6*0 = 0.
    cert = _rational_identity_cert(
        [(1, 30), (6, -15), (15, 5), (20, 0), (15, -1), (6, 0)], 0
    )
    src = generate_obligation(cert)
    assert src is not None
    assert "eq_of_mem_point" in src  # the new equality lemma
    assert "theorem obligation" in src
    assert "= 0 := by" in src


def test_generate_rational_identity_none_when_malformed() -> None:
    # Missing rhs / empty terms / non-integer data -> no obligation.
    assert generate_obligation(
        make_certificate(claim="x", payload={"type": "rational_identity", "lhs_terms": []}, honesty={})
    ) is None
    bad = make_certificate(
        claim="x",
        payload={"type": "rational_identity", "lhs_terms": [[1, 2]], "rhs": 1.5},
        honesty={},
    )
    assert generate_obligation(bad) is None


def test_generate_ln_chain_closure_and_format_obligations() -> None:
    chain = make_certificate(
        claim="finite LN chain closure",
        payload={
            "type": "ln_chain_closure",
            "identities": [
                {"lhs": [2, 3], "rhs": [4, 6]},
                {"lhs": [-5, 7], "rhs": [5, -7]},
            ],
        },
        honesty={},
        meta={"transcend_backend": "not_used"},
    )
    chain_src = generate_obligation(chain)
    assert chain_src is not None
    assert "allRatEq" in chain_src
    assert "no physical Dulac/LN membership" in chain_src

    fmt = make_certificate(
        claim="finite LN format bound",
        payload={
            "type": "ln_format_bound",
            "inequalities": [
                {"lhs": [2, 3], "rhs": [3, 4]},
                {"lhs": [-1, 2], "rhs": [0, 1]},
            ],
        },
        honesty={},
        meta={"transcend_backend": "not_used"},
    )
    fmt_src = generate_obligation(fmt)
    assert fmt_src is not None
    assert "allRatLt" in fmt_src


def test_ln_finite_obligations_refuse_false_malformed_and_tampered_data() -> None:
    false_chain = make_certificate(
        claim="false closure",
        payload={
            "type": "ln_chain_closure",
            "identities": [{"lhs": [1, 2], "rhs": [2, 3]}],
        },
        honesty={},
    )
    assert generate_obligation(false_chain) is None
    malformed_format = make_certificate(
        claim="malformed format",
        payload={
            "type": "ln_format_bound",
            "inequalities": [{"lhs": [1, 0], "rhs": [2, 1]}],
        },
        honesty={},
    )
    assert generate_obligation(malformed_format) is None

    valid = make_certificate(
        claim="finite LN format bound",
        payload={
            "type": "ln_format_bound",
            "inequalities": [{"lhs": [1, 3], "rhs": [1, 2]}],
        },
        honesty={},
    )
    tampered = dict(valid)
    tampered["payload"] = {
        "type": "ln_format_bound",
        "inequalities": [{"lhs": [2, 3], "rhs": [1, 2]}],
    }
    result = check_certificate(tampered)
    assert result.verified is False
    assert result.obligation == ""
    assert "digest" in result.detail


def test_generate_integer_matrix_syzygy_obligation() -> None:
    cert = make_certificate(
        claim="exact matrix syzygy",
        payload={
            "type": "integer_matrix_syzygy",
            "matrix": [[1, 2, 3], [4, 5, 6]],
            "vector": [1, -2, 1],
        },
        honesty={},
    )
    src = generate_obligation(cert)
    assert src is not None
    assert "def matrixSyzygy" in src
    assert "dotInt row vector == 0" in src
    assert "by\n  decide" in src


def test_generate_polynomial_identity_q_obligation() -> None:
    cert = make_certificate(
        claim="exact polynomial identity and margins",
        payload={
            "type": "polynomial_identity_q",
            "equations": [
                {"terms": [[2, 3], [-1, 1]], "rhs": 5},
                {"terms": [[7, 0]], "rhs": 0},
            ],
            "positive": [[1, 2], [9, 7]],
        },
        honesty={},
    )
    src = generate_obligation(cert)
    assert src is not None
    assert "def polynomialIdentity" in src
    assert "a * b + productSum rest" in src
    assert "allRatLt" in src
    assert "by\n  decide" in src

    malformed = make_certificate(
        claim="malformed polynomial identity",
        payload={
            "type": "polynomial_identity_q",
            "equations": [],
            "positive": [],
        },
        honesty={},
    )
    assert generate_obligation(malformed) is None


def test_polynomial_identity_q_round_trip_when_available() -> None:
    true_cert = make_certificate(
        claim="true polynomial identity",
        payload={
            "type": "polynomial_identity_q",
            "equations": [{"terms": [[2, 3], [-1, 1]], "rhs": 5}],
            "positive": [[1, 2]],
        },
        honesty={},
    )
    false_cert = make_certificate(
        claim="false polynomial identity",
        payload={
            "type": "polynomial_identity_q",
            "equations": [{"terms": [[2, 3]], "rhs": 7}],
            "positive": [[1, 2]],
        },
        honesty={},
    )
    true_result = check_certificate(true_cert)
    false_result = check_certificate(false_cert)
    if lean_check_available():  # pragma: no cover - Lean-equipped environment only
        assert true_result.verified
        assert not false_result.verified
    else:
        assert not true_result.available and not true_result.verified
        assert not false_result.available and not false_result.verified


def test_box_cover_tiling_round_trip_when_available() -> None:
    certificate = make_certificate(
        claim="exact rational box cover",
        payload={
            "type": "box_cover_tiling",
            "parent": [[[0, 1], [1, 1]], [[-1, 1], [1, 1]]],
            "uniform_bound": 2,
            "tree": {
                "kind": "split",
                "axis": 0,
                "cut": [1, 2],
                "left": {
                    "kind": "leaf",
                    "box": [[[0, 1], [1, 2]], [[-1, 1], [1, 1]]],
                    "count": 2,
                },
                "right": {
                    "kind": "leaf",
                    "box": [[[1, 2], [1, 1]], [[-1, 1], [1, 1]]],
                    "count": 1,
                },
            },
        },
        honesty={},
    )
    src = generate_obligation(certificate)
    assert src is not None
    assert "inductive CoverTree" in src
    assert "expected.set axis" in src
    assert "count ≤ bound" in src
    result = check_certificate(certificate)
    if lean_check_available():  # pragma: no cover - Lean-equipped environment only
        assert result.verified
    else:
        assert not result.available and not result.verified

    malformed = make_certificate(
        claim="malformed rational box cover",
        payload={
            "type": "box_cover_tiling",
            "parent": [],
            "uniform_bound": 2,
            "tree": {"kind": "leaf", "box": [], "count": 2},
        },
        honesty={},
    )
    assert generate_obligation(malformed) is None


def test_generate_winding_integer_isolation_obligation() -> None:
    cert = make_certificate(
        claim="unique winding integer",
        payload={
            "type": "winding_integer_isolation",
            "lo": 1.9,
            "hi": 2.1,
            "integer": 2,
        },
        honesty={},
    )
    src = generate_obligation(cert)
    assert src is not None
    assert "unique integer" in src
    assert "def windingIsolated" in src
    assert "integer - 1" in src
    assert "by\n  decide" in src


def test_new_finite_obligations_round_trip_when_available() -> None:
    matrix_cert = make_certificate(
        claim="exact matrix syzygy",
        payload={
            "type": "integer_matrix_syzygy",
            "matrix": [[1, 2, 3], [4, 5, 6]],
            "vector": [1, -2, 1],
        },
        honesty={},
    )
    winding_cert = make_certificate(
        claim="unique winding integer",
        payload={
            "type": "winding_integer_isolation",
            "lo": 1.9,
            "hi": 2.1,
            "integer": 2,
        },
        honesty={},
    )
    matrix_result = check_certificate(matrix_cert)
    winding_result = check_certificate(winding_cert)
    if lean_check_available():  # pragma: no cover - Lean-equipped environment only
        assert matrix_result.verified
        assert winding_result.verified
    else:
        assert not matrix_result.available and not matrix_result.verified
        assert not winding_result.available and not winding_result.verified


def test_rational_identity_round_trip_when_available() -> None:
    # A true identity earns a genuine kernel pass (only when lake is present); a
    # false one is rejected. Without a toolchain the bridge degrades gracefully.
    true_cert = _rational_identity_cert([(1, 12), (-1, 12)], 0)  # 12 - 12 = 0
    false_cert = _rational_identity_cert([(1, 1)], 0)  # 1 != 0
    true_result = check_certificate(true_cert)
    assert "eq_of_mem_point" in true_result.obligation
    if lean_check_available():  # pragma: no cover - Lean-equipped environment only
        assert true_result.verified is True
        assert check_certificate(false_cert).verified is False
    else:
        assert true_result.available is False and true_result.verified is False


def test_generate_returns_none_for_unsupported() -> None:
    assert generate_obligation({"foo": "bar"}) is None
    # An interval that straddles zero has no sign obligation.
    assert generate_obligation(interval_certificate("q", Interval(-1.0, 1.0))) is None


def test_ratio_at_or_above_one_is_not_a_gap_obligation() -> None:
    assert generate_obligation({"subdominant_ratio_upper": 1.0}) is None
    assert generate_obligation({"subdominant_ratio_upper": 1.5}) is None


def test_tampered_certificate_is_rejected_before_lean() -> None:
    cert = seal_certificate(
        {"subdominant_ratio_upper": 0.5, "honesty": {"unproven_claim": False}}
    )
    tampered = dict(cert)
    tampered["subdominant_ratio_upper"] = 0.1  # body changed, digest stale
    result = check_certificate(tampered)
    assert result.verified is False
    assert "digest" in result.detail


def test_unsealed_certificate_is_refused_before_lean() -> None:
    # A payload with no digest at all must be refused exactly like a forged one:
    # accepting it would let an unsigned mapping earn verified=True, so the seal
    # is required rather than merely checked-when-present.
    unsealed = {"subdominant_ratio_upper": 0.5, "honesty": {"unproven_claim": False}}
    assert generate_obligation(unsealed) is not None  # the obligation itself is fine
    result = check_certificate(unsealed)
    assert result.verified is False
    assert result.obligation == ""  # refused before any Lean was emitted
    assert "unsealed" in result.detail
    # Sealing the very same body makes it acceptable to the bridge again.
    assert generate_obligation(seal_certificate(unsealed)) is not None
    assert check_certificate(seal_certificate(unsealed)).obligation != ""


def test_check_certificate_graceful_without_toolchain() -> None:
    cert = seal_certificate(
        {"subdominant_ratio_upper": 0.625, "honesty": {"unproven_claim": False}}
    )
    result = check_certificate(cert)
    # Whatever the environment, the obligation is generated and no exception is raised.
    assert "gapNumerator 5 8" in result.obligation
    if not lean_check_available():
        assert result.available is False
        assert result.verified is False
    else:  # pragma: no cover - only on a machine with Lean installed
        assert result.available is True
        assert result.verified is True


def _perron_prover() -> FunctionProver:
    def prove(conjecture: Conjecture) -> ProofAttempt:
        ratio = float(conjecture.data["ratio"])
        cert: Certificate = seal_certificate(
            {
                "schema_version": "1.0",
                "subdominant_ratio_upper": ratio,
                "spectral_gap_lower": 1.0 - ratio,
                "honesty": {"unproven_claim": False},
            }
        )
        return ProofAttempt(status="PROVED", certificate=cert)

    return FunctionProver(name="perron", kinds=frozenset({"perron"}), prove_fn=prove)


def test_machine_formal_claim_gate() -> None:
    machine = ProofMachine().register(_perron_prover())
    conj = Conjecture(
        "gap", "perron", {"ratio": 0.625}, claims={"theorem_prover_verified": True}
    )
    verdict = machine.evaluate(conj)
    if lean_check_available():  # pragma: no cover - Lean-equipped environment only
        assert verdict.status == "PROVED"
        assert verdict.theorem_prover_verified is True
    else:
        # Asserting a formal claim with no kernel available downgrades to BLOCKED.
        assert verdict.status == "BLOCKED"
        assert verdict.theorem_prover_verified is False
        assert any("theorem_prover_verified" in o for o in verdict.obligations)


def test_machine_without_formal_claim_is_unaffected() -> None:
    machine = ProofMachine().register(_perron_prover())
    verdict = machine.evaluate(Conjecture("gap", "perron", {"ratio": 0.625}))
    # No formal claim, no lean_check flag -> stays PROVED, flag stays False.
    assert verdict.status == "PROVED"
    assert verdict.theorem_prover_verified is False


def test_machine_lean_check_flag_does_not_block_unclaimed() -> None:
    machine = ProofMachine().register(_perron_prover())
    verdict = machine.evaluate(
        Conjecture("gap", "perron", {"ratio": 0.625}), lean_check=True
    )
    # lean_check requested but no claim asserted: never blocks; flag mirrors the kernel.
    assert verdict.status == "PROVED"
    assert verdict.theorem_prover_verified is lean_check_available()
