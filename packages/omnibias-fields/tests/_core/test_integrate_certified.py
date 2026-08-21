# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 03-06: certified field integration refuses without a derivative bound."""

from __future__ import annotations

import pytest
from omnibias.core.cubature import MomentSystem, apply_rule, solve_rule
from omnibias.core.verified.interval import Interval
from omnibias.fields._core.quadrature import integrate_certified


def test_integrate_certified_contains_x4() -> None:
    rule = solve_rule(MomentSystem.lebesgue(6), nodes=2, free_nodes=True)
    values = [x**4 for x in rule.nodes]
    enc = integrate_certified(values, rule, deriv_bound=Interval.point(24.0), degree=3)
    exact = 0.4
    assert enc.contains(exact)
    main = apply_rule(rule, lambda x: x**4)
    assert enc.contains(main)


def test_integrate_certified_refuses() -> None:
    rule = solve_rule(MomentSystem.lebesgue(4), nodes=2, free_nodes=True)
    with pytest.raises(ValueError, match="refuses"):
        integrate_certified([0.0, 0.0], rule, deriv_bound=None)
