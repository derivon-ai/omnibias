# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the :class:`OperatorMultiBiasUnit` primitive."""

from __future__ import annotations

import pytest
import torch
from omnibias.torch import ActivationSpec, OperatorMultiBiasUnit
from omnibias.torch.activations import list_activations


def test_construction_defaults() -> None:
    ombu = OperatorMultiBiasUnit(num_channels=4, K=2, base="sigmoid")
    assert ombu.num_channels == 4
    assert ombu.K == 2
    assert ombu.spec.name == "sigmoid"
    assert ombu.is_identity_nested
    assert ombu.biases.shape == (4, 2)
    assert ombu.signs.shape == (4, 2)


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6])
def test_forward_output_shape(K: int) -> None:
    C = 7
    ombu = OperatorMultiBiasUnit(num_channels=C, K=K, base="sigmoid")
    z = torch.randn(3, C)
    y = ombu(z)
    assert y.shape == (3, C)


def test_invalid_input_shape_raises() -> None:
    ombu = OperatorMultiBiasUnit(num_channels=4, K=2, base="sigmoid")
    with pytest.raises(ValueError):
        ombu(torch.randn(3, 5))


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "softplus", "gaussian", "exp"])
def test_analytic_derivative_first_order_matches_spec(name: str) -> None:
    """For K=2 with tied biases, ``analytic_derivative(z, 1)`` equals the
    spec's first-derivative function evaluated at ``z + bias_mean``."""
    ombu = OperatorMultiBiasUnit(num_channels=3, K=2, base=name, init_bias=0.5)
    z = torch.linspace(-2, 2, 16).unsqueeze(-1).expand(16, 3).contiguous()
    out = ombu.analytic_derivative(z, order=1)
    ref = ombu.spec.derivative(z + 0.5)
    assert torch.allclose(out, ref, atol=1e-5)


def test_analytic_derivative_no_fastpath_raises() -> None:
    """A base spec without a fast-path kernel raises ``NotImplementedError``.

    ``relu`` (and the rest of the registry) now carry a closed-form
    almost-everywhere tower, so construct a deliberately fastpath-less spec to
    exercise the guard.
    """
    no_fastpath = ActivationSpec(
        name="_no_fastpath_probe",
        forward=torch.relu,
        derivative=None,
        fastpath=None,
    )
    ombu = OperatorMultiBiasUnit(num_channels=2, K=3, base=no_fastpath)
    with pytest.raises(NotImplementedError):
        ombu.analytic_derivative(torch.randn(4, 2), order=2)


def test_relu_analytic_derivative_second_order_is_zero() -> None:
    """relu now exposes an all-orders a.e. tower: order >= 2 is 0 everywhere.

    (Previously order >= 2 raised; this pins the deliberate behavioral change.)
    """
    ombu = OperatorMultiBiasUnit(num_channels=2, K=3, base="relu")
    out = ombu.analytic_derivative(torch.randn(4, 2), order=2)
    assert torch.equal(out, torch.zeros_like(out))


def test_learnable_signs_become_parameter() -> None:
    ombu = OperatorMultiBiasUnit(num_channels=2, K=2, base="sigmoid", learnable_signs=True)
    assert isinstance(ombu.signs, torch.nn.Parameter)
    assert ombu.signs.requires_grad


def test_buffer_signs_stay_buffer() -> None:
    ombu = OperatorMultiBiasUnit(num_channels=2, K=2, base="sigmoid", learnable_signs=False)
    assert not isinstance(ombu.signs, torch.nn.Parameter)


def test_custom_biases_init_validates_shape() -> None:
    with pytest.raises(ValueError):
        OperatorMultiBiasUnit(num_channels=3, K=2, base="sigmoid", biases_init=torch.zeros(3, 4))


def test_custom_signs_init_broadcast_from_K() -> None:
    """Pass a ``(K,)`` signs init and confirm it broadcasts to ``(C, K)``."""
    s = torch.tensor([0.5, 0.5])
    ombu = OperatorMultiBiasUnit(num_channels=3, K=2, base="sigmoid", signs_init=s)
    assert ombu.signs.shape == (3, 2)
    assert torch.allclose(ombu.signs, s.unsqueeze(0).expand(3, 2))


@pytest.mark.parametrize("name", list_activations())
def test_every_registered_activation_can_be_constructed(name: str) -> None:
    ombu = OperatorMultiBiasUnit(num_channels=2, K=2, base=name)
    z = torch.randn(3, 2)
    out = ombu(z)
    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()
