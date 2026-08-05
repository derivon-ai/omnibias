# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Lean-certified hypergeometric identity proofs (creative telescoping + rational_identity)."""

from __future__ import annotations

from fractions import Fraction
from math import comb

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.proof.lean_check import generate_obligation, lean_check_available
from omnibias.holonomic._core.certify import (
    prove_hypergeometric_identity,
    prove_identity_zeilberger,
)
from omnibias.holonomic._core.hyperterm import ProperTerm, binomial_nk

# A representative spread of classic identities (>= 8) the guesser + prover handle.
IDENTITIES = [
    ("sum C(n,k) = 2^n", lambda n, k: comb(n, k), lambda n: 2**n, None),
    ("sum C(n,k)^2 = C(2n,n)", lambda n, k: comb(n, k) ** 2, lambda n: comb(2 * n, n), None),
    ("sum k C(n,k) = n 2^{n-1}", lambda n, k: k * comb(n, k), lambda n: n * 2 ** (n - 1) if n else 0, None),
    ("sum k^2 C(n,k)", lambda n, k: k * k * comb(n, k), lambda n: n * (n + 1) * 2 ** (n - 2) if n else 0, None),
    ("Vandermonde C(n,k)C(n,n-k)", lambda n, k: comb(n, k) * comb(n, n - k), lambda n: comb(2 * n, n), None),
    ("sum_{k<=n} k = n(n+1)/2", lambda n, k: k, lambda n: n * (n + 1) // 2, None),
    ("sum k^2 = n(n+1)(2n+1)/6", lambda n, k: k * k, lambda n: n * (n + 1) * (2 * n + 1) // 6, None),
    ("sum k^3 = (n(n+1)/2)^2", lambda n, k: k**3, lambda n: (n * (n + 1) // 2) ** 2, None),
    ("hockey-stick C(k,1) = C(n+1,2)", lambda n, k: comb(k, 1), lambda n: comb(n + 1, 2), None),
    ("sum 2^k = 2^{n+1}-1", lambda n, k: 2**k, lambda n: 2 ** (n + 1) - 1, None),
]


@pytest.mark.parametrize("name,summand,closed,kb", IDENTITIES)
def test_identity_is_proven_on_range(name, summand, closed, kb) -> None:  # type: ignore[no-untyped-def]
    proof = prove_hypergeometric_identity(
        name=name, summand=summand, closed_form=closed, n_max=14, k_bound=kb
    )
    assert proof.identity_holds_on_range
    assert proof.order >= 1
    assert len(proof.certificates) > 0
    assert proof.certificates_sealed
    assert proof.obligations_generated


def test_at_least_eight_distinct_identities() -> None:
    assert len({name for name, *_ in IDENTITIES}) >= 8


def test_false_identity_is_refuted() -> None:
    # Wrong closed form (scaled by 2) must NOT be reported as holding.
    proof = prove_hypergeometric_identity(
        name="bogus", summand=lambda n, k: comb(n, k), closed_form=lambda n: 2 * 2**n, n_max=10
    )
    assert not proof.identity_holds_on_range


def test_obligations_are_rational_identities_summing_to_zero() -> None:
    proof = prove_hypergeometric_identity(
        name="sum C(n,k)^2 = C(2n,n)",
        summand=lambda n, k: comb(n, k) ** 2,
        closed_form=lambda n: comb(2 * n, n),
        n_max=10,
    )
    for cert in proof.certificates:
        payload = cert["payload"]
        assert payload["type"] == "rational_identity"
        assert payload["rhs"] == 0
        # the exact Int identity the Lean kernel checks: sum_i c_i m_i = 0.
        assert sum(c * m for c, m in payload["lhs_terms"]) == 0
        assert generate_obligation(cert) is not None


def test_certificates_have_value_and_recurrence_obligations() -> None:
    proof = prove_hypergeometric_identity(
        name="sum C(n,k) = 2^n",
        summand=lambda n, k: comb(n, k),
        closed_form=lambda n: 2**n,
        n_max=12,
    )
    kinds = {c["meta"]["kind"] for c in proof.certificates}
    assert "value_equality" in kinds
    assert "recurrence_on_sum" in kinds
    assert "recurrence_on_closed_form" in kinds


def test_rational_valued_identity() -> None:
    # sum_{k=0}^{n} C(n,k)/(k+1) = (2^{n+1} - 1)/(n + 1): exercises rational cross-multiply.
    proof = prove_hypergeometric_identity(
        name="sum C(n,k)/(k+1) = (2^{n+1}-1)/(n+1)",
        summand=lambda n, k: Fraction(comb(n, k), k + 1),
        closed_form=lambda n: Fraction(2 ** (n + 1) - 1, n + 1),
        n_max=12,
    )
    assert proof.identity_holds_on_range
    assert proof.certificates_sealed


def test_theorem_prover_flag_reflects_lean_availability() -> None:
    proof = prove_hypergeometric_identity(
        name="sum C(n,k) = 2^n",
        summand=lambda n, k: comb(n, k),
        closed_form=lambda n: 2**n,
        n_max=8,
        prove_lean=False,
    )
    # Never forged: no Lean run -> flag is False.
    assert not proof.theorem_prover_verified


@pytest.mark.skipif(not lean_check_available(), reason="Lean toolchain / kernel not present")
def test_lean_kernel_accepts_obligations() -> None:  # pragma: no cover - env dependent
    proof = prove_hypergeometric_identity(
        name="sum C(n,k)^2 = C(2n,n)",
        summand=lambda n, k: comb(n, k) ** 2,
        closed_form=lambda n: comb(2 * n, n),
        n_max=6,
        prove_lean=True,
        lean_sample=4,
    )
    assert all(r.verified for r in proof.lean_results)
    assert all(verify_certificate_digest(c) for c in proof.certificates)


def test_lean_degrades_gracefully_when_absent() -> None:
    proof = prove_hypergeometric_identity(
        name="sum C(n,k) = 2^n",
        summand=lambda n, k: comb(n, k),
        closed_form=lambda n: 2**n,
        n_max=6,
        prove_lean=True,
        lean_sample=2,
    )
    if not proof.lean_available:
        assert all(not r.available for r in proof.lean_results)
        assert not proof.theorem_prover_verified


# --------------------------------------------------------------------------- #
# All-n certified identities (true Zeilberger + Lean grid obligations).
# --------------------------------------------------------------------------- #
def _pow2() -> ProperTerm:
    return ProperTerm((), geom_n=Fraction(2))


def _central() -> ProperTerm:
    return ProperTerm(((2, 0, 0, 1), (1, 0, 0, -2)))


def test_zeilberger_all_n_row_sum() -> None:
    proof = prove_identity_zeilberger(name="row_sum", term=binomial_nk(), closed_form_term=_pow2())
    assert proof.all_n
    assert proof.identity_holds_on_range
    assert proof.certificates_sealed
    assert proof.obligations_generated
    assert "PROVEN(all n)" in proof.pretty()


def test_zeilberger_all_n_central_binomial() -> None:
    proof = prove_identity_zeilberger(
        name="central", term=binomial_nk().power(2), closed_form_term=_central()
    )
    assert proof.all_n
    assert proof.order == 1
    kinds = {c["meta"]["kind"] for c in proof.certificates}
    assert "telescoping_grid" in kinds
    assert "recurrence_on_closed_form" in kinds
    assert "initial_condition" in kinds


def test_zeilberger_grid_obligations_sum_to_zero() -> None:
    proof = prove_identity_zeilberger(name="row_sum", term=binomial_nk(), closed_form_term=_pow2())
    grid = [c for c in proof.certificates if c["meta"]["kind"] == "telescoping_grid"]
    assert len(grid) > 0
    for cert in grid:
        payload = cert["payload"]
        assert payload["type"] == "rational_identity"
        assert sum(c * m for c, m in payload["lhs_terms"]) == 0
        assert cert["meta"]["all_n"] is True


def test_zeilberger_all_n_beats_range_proof() -> None:
    # Baseline: the guess-based proof certifies only a bounded range.
    ranged = prove_hypergeometric_identity(
        name="row_sum", summand=lambda n, k: comb(n, k), closed_form=lambda n: 2**n, n_max=12
    )
    alln = prove_identity_zeilberger(name="row_sum", term=binomial_nk(), closed_form_term=_pow2())
    assert not ranged.all_n
    assert alln.all_n


def test_zeilberger_false_identity_not_all_n() -> None:
    # sum_k C(n,k) = C(2n,n) is FALSE: the closed form does not obey the row-sum recurrence.
    proof = prove_identity_zeilberger(
        name="bogus", term=binomial_nk(), closed_form_term=_central()
    )
    assert not proof.all_n
    assert not proof.identity_holds_on_range
