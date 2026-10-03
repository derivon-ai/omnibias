# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Uniform comparison GRAZING E_sigma on eps in [0, 1/8]; not Lohner or G1."""

from __future__ import annotations

from fractions import Fraction

from omnibias.dynamics.e_sigma_from0 import height_majorant
from omnibias.dynamics.e_sigma_unif import (
    I_cancel,
    enclose_uniform,
    identity_verdicts,
    report,
    residual_de_eps0,
    residual_E_eps0,
    residual_I_cancel,
    residual_I_limit,
)
from omnibias.dynamics.hilbert16_ledger import default_h16_ledger, derived_parent_flags


def test_e_sigma_unif_identities_are_exact() -> None:
    vstar, eps = Fraction(6, 5), Fraction(1, 16)
    rho, sigma, c = Fraction(1, 4), Fraction(-1), Fraction(2)
    assert residual_I_cancel(vstar, eps) == 0
    assert residual_I_limit(vstar) == 0
    assert residual_E_eps0(vstar, vstar * vstar / 2, sigma, rho, c) == 0
    assert residual_de_eps0(rho, vstar, Fraction(7, 10)) == 0
    assert I_cancel(vstar, eps) == height_majorant(vstar, eps) - 4 * eps**3
    assert I_cancel(vstar, Fraction(0)) == vstar * vstar / 2
    assert all(status == "PROVED" for status in identity_verdicts().values())
    coarse = enclose_uniform(n_slabs=1)
    assert coarse["all_positive"] is False


def test_e_sigma_unif_hit_does_not_pass_g1() -> None:
    payload = report().to_payload()
    assert payload["schema"] == "hilbert16-e-sigma-unif-v1"
    assert payload["e_sigma_unif_hit"] is True
    assert payload["outgoing_first_hit"] is False
    assert payload["uniform"]["n_slabs"] == 8
    assert payload["uniform"]["all_positive"] is True
    assert payload["coarse"]["all_positive"] is False
    assert payload["refuse_v1"]["all_positive"] is False
    honesty = payload["honesty"]
    assert honesty["e_sigma_unif_hit"] is True
    assert honesty["outgoing_first_hit"] is False
    assert honesty["g1_passed"] is False
    assert honesty["full_hilbert16_solved"] is False
    reasons = honesty["_honesty_reasons"]
    assert reasons["g1_passed"]["reason"] == "gate_open"
    assert "e_sigma_unif_hit" not in reasons


def test_local_e_sigma_unif_never_earns_a_parent_flag() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_e_sigma_unif")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
