# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Apolar reduction checked against an independently differentiated function."""

import pytest
import torch
from omnibias.core.operator_spec import OperatorSpec
from omnibias.torch.experimental._operator_apolar import apolar_derivative, select_apolar
from omnibias.torch.experimental._operator_ir import (
    NotStructured,
    add,
    constant,
    multiply,
    unary,
    variable,
)
from omnibias.torch.experimental._operator_lowering import Lowering


@pytest.mark.parametrize("power", [1, 2])
def test_overlapping_expression_quotient_values_and_parameter_gradients(power):
    x = torch.tensor([.2, -.3, .4], dtype=torch.float64)
    parameter = torch.tensor(.7, dtype=x.dtype, requires_grad=True)
    a, b, c = (variable(value, i) for i, value in enumerate(x))
    ab = multiply(a, b)
    first = unary(ab, torch.sin(ab.value), "sin")
    inner = multiply(add(add(a, b), c), constant(parameter))
    second = unary(inner, torch.tanh(inner.value), "tanh")
    expr = multiply(add(first, second), add(a, c))
    request = OperatorSpec.laplacian(power)
    algebra = select_apolar(request, 3)
    actual = apolar_derivative(expr, algebra, Lowering(request, "recurrence"))

    def primal(z):
        return (torch.sin(z[0]*z[1]) + torch.tanh(parameter*z.sum())) * (z[0]+z[2])
    function = primal
    for _ in range(power):
        previous = function
        def function(z, f=previous):
            return torch.func.hessian(f)(z).trace()
    reference = function(x)
    torch.testing.assert_close(actual, reference, rtol=3e-11, atol=3e-11)
    actual_gradient = torch.autograd.grad(actual, parameter, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference, parameter)[0]
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=3e-11, atol=3e-11)
    assert algebra.size == (5 if power == 1 else 14)


def test_declines_nonhomogeneous_and_unhelpful_queries():
    with pytest.raises(NotStructured, match="homogeneous"):
        select_apolar(OperatorSpec.polynomial({(0, 0): 1, (2, 0): 1}), 2)
    with pytest.raises(NotStructured, match="not smaller"):
        select_apolar(OperatorSpec.mixed((2, 1)), 2)
    with pytest.raises(NotStructured, match="planning declined"):
        select_apolar(OperatorSpec.mixed((1,)*24), 24)
