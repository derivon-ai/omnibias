# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 03-06: neural quadrature / cubature, gates G1–G5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omnibias.core.cubature import (
    MomentSystem,
    apply_rule,
    certified_error,
    design_rule,
    dimension_scaling_table,
    honesty_payload,
    pack_moment,
    solve_rule,
    tensor_product_cost,
)
from omnibias.core.multipack import PackSpec
from omnibias.core.verified.interval import Interval


def test_g1_classical_recovery() -> None:
    system = MomentSystem.lebesgue(24)
    for n in range(2, 13):
        rule = solve_rule(system, nodes=n, free_nodes=True, functional="point")
        xs, ws = np.polynomial.legendre.leggauss(n)
        np.testing.assert_allclose(sorted(rule.nodes), sorted(xs), atol=1e-13, rtol=0.0)
        got = np.array(sorted(rule.weights), dtype=np.float64)
        ref = np.array(sorted(ws), dtype=np.float64)
        np.testing.assert_allclose(got, ref, atol=1e-13, rtol=0.0)


def test_g2_moment_exactness_point_and_pack() -> None:
    system = MomentSystem.lebesgue(8)
    rule = solve_rule(system, nodes=3, free_nodes=True)
    for j in range(rule.degree + 1):
        got = rule.apply_monomial(j)
        exact = system.target_moments[j]
        assert abs(got - exact) <= 1e-13 * max(1.0, abs(exact)), (j, got, exact)
    pack = solve_rule(system, nodes=2, free_nodes=True, functional="pack", pack_scale=0.1, pack_base="gaussian")
    assert pack.bias_cancelled
    for j in (0, 2):
        got = pack.apply_monomial(j)
        exact = system.target_moments[j]
        assert abs(got - exact) <= 1e-13 * max(1.0, abs(exact)), (j, got, exact)
    # Uncorrected GL nodes with the same bump miss m2 by ~2 var.
    var = pack_moment(PackSpec(order=0, mean=0.0), 2, scale=0.1, base="gaussian")
    assert var == pytest.approx(0.01, rel=1e-12)


def test_g3_certified_error_sound_and_attained() -> None:
    system = MomentSystem.lebesgue(6)
    rule = solve_rule(system, nodes=2, free_nodes=True)
    exact = 2.0 / 5.0
    got = apply_rule(rule, lambda x: x**4)
    true_err = got - exact
    assert abs(true_err + 0.1777777777777778) < 1e-12 or abs(true_err - 0.1777777777777778) < 1e-12
    bound = Interval.point(24.0)
    enc = certified_error(rule, deriv_bound=bound, degree=3)
    assert enc.contains(true_err)
    assert enc.contains(-true_err)
    # Attained to within a small factor on the extremal constant-f^(4) case.
    assert abs(true_err) >= 0.4 * enc.mag
    with pytest.raises(ValueError, match="refuses"):
        certified_error(rule, deriv_bound=None, degree=3)


def test_g4_design_win() -> None:
    def family(a: float) -> tuple[object, float]:
        def fn(x: float, coeff: float = a) -> float:
            return x**4 + coeff * x**6

        return fn, 2.0 / 5.0 + a * (2.0 / 7.0)

    seeds = (0.05, 0.07, 0.09, 0.11, 0.13)
    gl = solve_rule(MomentSystem.lebesgue(6), nodes=2, free_nodes=True)
    integrands = []
    exacts = []
    for a in seeds:
        fn, exact = family(a)
        integrands.append(fn)
        exacts.append(exact)
    designed = design_rule(integrands, exacts, nodes=2)
    ratios = []
    for fn, exact in zip(integrands, exacts, strict=True):
        e_gl = abs(apply_rule(gl, fn) - exact)
        e_des = abs(apply_rule(designed, fn) - exact)
        ratios.append(e_gl / max(e_des, 1e-18))
    assert float(np.mean(ratios)) >= 10.0
    assert all(r > 1.0 for r in ratios)


def test_g5_honest_scope() -> None:
    table = dimension_scaling_table(nodes_per_axis=8, max_dim=6)
    assert table[0] == (1, 8)
    assert table[3] == (4, 8**4)
    # Tensor product is prohibitive once n^D exceeds 4096 at this n.
    prohibitive = min(d for d, cost in table if cost > 4096)
    assert prohibitive == 5
    assert tensor_product_cost(nodes_per_axis=8, dim=5) == 32768
    payload = honesty_payload()
    assert payload["non_product_cubature"] is False
    assert payload["temperature_collapse"] is False
    root = Path(__file__).resolve().parents[1] / "src" / "omnibias" / "core" / "cubature.py"
    text = root.read_text(encoding="utf-8")
    assert "founding bias collapse" in text
    assert "delta -> 0" in text
    assert "beta -> inf" in text
    assert "feasibility" in text
    assert "do not conflate" in text.lower()
