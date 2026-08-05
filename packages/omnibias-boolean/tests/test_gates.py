# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Soft logic gates (torch): exactness on cube vertices and differentiability."""

from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")

from omnibias.boolean.torch.ops import gates  # noqa: E402

TWO = [
    (gates.soft_and, lambda a, b: a & b),
    (gates.soft_or, lambda a, b: a | b),
    (gates.soft_xor, lambda a, b: a ^ b),
    (gates.soft_nand, lambda a, b: 1 - (a & b)),
    (gates.soft_nor, lambda a, b: 1 - (a | b)),
    (gates.soft_xnor, lambda a, b: 1 - (a ^ b)),
    (gates.soft_implies, lambda a, b: 0 if (a == 1 and b == 0) else 1),
]


@pytest.mark.parametrize(("soft", "ref"), TWO)
def test_two_input_gates_exact_on_vertices(soft, ref) -> None:  # type: ignore[no-untyped-def]
    for a, b in itertools.product((0, 1), repeat=2):
        out = float(soft(torch.tensor(float(a)), torch.tensor(float(b))))
        assert out == pytest.approx(float(ref(a, b)), abs=1e-12)


def test_not_and_majority() -> None:
    assert float(gates.soft_not(torch.tensor(0.0))) == 1.0
    assert float(gates.soft_not(torch.tensor(1.0))) == 0.0
    for a, b, c in itertools.product((0, 1), repeat=3):
        out = float(
            gates.soft_majority3(
                torch.tensor(float(a)), torch.tensor(float(b)), torch.tensor(float(c))
            )
        )
        assert out == pytest.approx(float(1 if (a + b + c) >= 2 else 0), abs=1e-12)


def test_threshold_gates_hard_forward() -> None:
    for a, b in itertools.product((0, 1), repeat=2):
        ta, tb = torch.tensor(float(a)), torch.tensor(float(b))
        assert float(gates.threshold_and(ta, tb)) == float(a & b)
        assert float(gates.threshold_or(ta, tb)) == float(a | b)
        assert float(gates.threshold_not(ta)) == float(1 - a)


def test_soft_gate_is_differentiable() -> None:
    a = torch.tensor(0.3, requires_grad=True)
    b = torch.tensor(0.7, requires_grad=True)
    gates.soft_xor(a, b).backward()
    assert a.grad is not None and b.grad is not None


def test_threshold_gate_surrogate_gradient_nonzero() -> None:
    a = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    gates.threshold_and(a, b, beta=4.0).backward()
    # Hard forward, but the sigmoid surrogate still passes a gradient.
    assert abs(float(a.grad)) > 0.0
