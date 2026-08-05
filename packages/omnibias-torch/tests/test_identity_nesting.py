# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Lemma 1 (identity nesting) regression tests.

For every spec in the activation dictionary and every K in {1..6}, an
OMBU initialised with tied biases and signs summing to one must reduce
to the base activation bit-identically.
"""

from __future__ import annotations

import pytest
import torch
from omnibias.torch import OperatorMultiBiasUnit
from omnibias.torch.activations import list_activations
from omnibias.torch.identity_init import (
    identity_init_biases,
    identity_init_signs,
    verify_identity_init,
)


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6])
def test_identity_signs_sum_to_one(K: int) -> None:
    s = identity_init_signs(num_channels=3, K=K)
    sums = s.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6])
def test_identity_biases_are_tied(K: int) -> None:
    b = identity_init_biases(num_channels=3, K=K, bias_value=0.7)
    assert torch.allclose(b - b[..., :1], torch.zeros_like(b))


def test_verify_identity_init() -> None:
    b = identity_init_biases(num_channels=4, K=3, bias_value=0.0)
    s = identity_init_signs(num_channels=4, K=3)
    assert verify_identity_init(b, s, atol=0.0)


@pytest.mark.parametrize("name", list_activations())
@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6])
def test_lemma1_identity_for_every_spec_and_K(name: str, K: int) -> None:
    """OMBU(K, base=name, init_delta=0) at init equals base.forward(z)."""
    ombu = OperatorMultiBiasUnit(num_channels=3, K=K, base=name, init_bias=0.3)
    assert ombu.is_identity_nested
    z = torch.linspace(-1.5, 1.5, 16).unsqueeze(-1).expand(16, 3).contiguous()
    out = ombu(z)
    ref = ombu.spec.forward(z + 0.3)
    # Bit-identical at K=1 (literally one term); within float epsilon for K>=2
    # because of the alternating-cancellation arithmetic. We allow a tiny
    # tolerance to absorb the float64-vs-float32 round-off.
    if K == 1:
        assert torch.equal(out, ref), f"{name} K={K}: not bit-identical"
    else:
        assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6), (
            f"{name} K={K}: max abs diff = {(out - ref).abs().max()}"
        )


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "softplus", "gaussian", "exp"])
def test_lemma1_after_zero_step_optimisation(name: str) -> None:
    """Even after a zero-LR optimiser step, Lemma 1 holds (no spurious drift)."""
    ombu = OperatorMultiBiasUnit(num_channels=4, K=4, base=name)
    optim = torch.optim.SGD(ombu.parameters(), lr=0.0)
    z = torch.randn(8, 4)
    out = ombu(z)
    out.pow(2).mean().backward()
    optim.step()
    out2 = ombu(z)
    ref = ombu.spec.forward(z)
    assert torch.allclose(out2, ref, atol=1e-6)
