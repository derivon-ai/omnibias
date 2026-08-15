# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tanh-method travelling waves G1/G2/G3/G5 (theory 02-09)."""

from __future__ import annotations

import math

from omnibias.core.tanh_method import (
    G1_NAMES,
    PDESpec,
    PDETerm,
    TermKind,
    balance_degree,
    classical_pdes,
    evaluate_ansatz,
    published_ansatz,
    solve_ansatz,
    substitute,
    verify_exact,
)


def test_g1_symbolic_exactness() -> None:
    pdes = classical_pdes()
    for name in G1_NAMES:
        pde = pdes[name]
        found = solve_ansatz(pde)
        assert found, name
        assert verify_exact(pde, found[0]), name
        assert all(c == 0 for c in substitute(pde, found[0])), name


def test_g2_balance_degree() -> None:
    pdes = classical_pdes()
    for name in G1_NAMES:
        pde = pdes[name]
        ans = published_ansatz(name)
        if ans.kind == "tanh_poly":
            assert balance_degree(pde) == ans.degree, name


def test_g3_numerical_residual() -> None:
    pdes = classical_pdes()
    for name in G1_NAMES:
        pde = pdes[name]
        ans = published_ansatz(name)
        mag = 0.0
        worst = 0.0
        for x in (-1.0, -0.3, 0.0, 0.5, 1.2):
            for t in (0.0, 0.25):
                xi = float(ans.wavenumber) * x - float(ans.frequency) * t
                tnh = math.tanh(xi)
                coeffs = substitute(pde, ans)
                res = 0.0
                p = 1.0
                for c in coeffs:
                    res += float(c) * p
                    p *= tnh
                mag = max(mag, abs(evaluate_ansatz(ans, x, t)), 1e-16)
                worst = max(worst, abs(res))
        assert worst <= 1e-14 * max(mag, 1.0), (name, worst, mag)


def test_g5_negative_control() -> None:
    heat = PDESpec(
        "heat",
        (PDETerm(TermKind.U_T, 1), PDETerm(TermKind.U_XX, -1)),
    )
    assert solve_ansatz(heat) == ()
