# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Selected SoftSort rows preserve values and the original backend Jacobian."""

from __future__ import annotations

import math

import numpy as np
import pytest

SCORES = [
    [3.0, -1.0, 4.0, 1.5, 9.0, 2.0],
    [3.0, 1.0, 3.0, 1.0, -2.0, -2.0],
    [2.0] * 6,
]


def _scalar_reference(scores: list[float], k: int, temperature: float) -> np.ndarray:
    """Independent scalar evaluation of the mathematical row distributions."""
    result = [0.0] * len(scores)
    for target in sorted(scores, reverse=True)[:k]:
        weights = [math.exp(-abs(target - score) / temperature) for score in scores]
        denominator = math.fsum(weights)
        result = [value + weight / denominator for value, weight in zip(result, weights, strict=True)]
    return np.asarray(result)


@pytest.mark.parametrize("scores", SCORES)
@pytest.mark.parametrize("k", [1, 3, 6])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_selected_rows_values_and_complete_jacobian(
    scores: list[float], k: int, dtype_name: str, backend: str
) -> None:
    temperature = 0.7
    tolerance = 2e-6 if dtype_name == "float32" else 2e-13
    reference = _scalar_reference(scores, k, temperature)
    if backend == "jax":
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        jax.config.update("jax_enable_x64", True)
        from omnibias.graph.jax.ops.relaxation import soft_sort_permutation, soft_top_k

        x = jnp.asarray(scores, dtype=getattr(jnp, dtype_name)).reshape(2, 3)

        def selected(value):
            return soft_top_k(value, k, temperature=temperature)

        def original(value):
            return soft_sort_permutation(value, temperature=temperature)[:k].sum(axis=0)

        values = np.asarray(jax.jit(selected)(x))
        new_jacobian = np.asarray(jax.jacrev(selected)(x))
        old_jacobian = np.asarray(jax.jacrev(original)(x))
    else:
        torch = pytest.importorskip("torch")
        from omnibias.graph.torch.ops.relaxation import soft_sort_permutation, soft_top_k

        x = torch.tensor(scores, dtype=getattr(torch, dtype_name)).reshape(2, 3)

        def selected(value):
            return soft_top_k(value, k, temperature=temperature)

        def original(value):
            return soft_sort_permutation(value, temperature=temperature)[:k].sum(dim=0)

        values = selected(x).detach().numpy()
        new_jacobian = torch.autograd.functional.jacobian(selected, x).numpy()
        old_jacobian = torch.autograd.functional.jacobian(original, x).numpy()
    np.testing.assert_allclose(values, reference, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(new_jacobian, old_jacobian, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(values.sum(), k, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("temperature", [0.0, -0.5])
def test_selected_rows_invalid_temperature(backend: str, temperature: float) -> None:
    if backend == "jax":
        jnp = pytest.importorskip("jax.numpy")
        from omnibias.graph.jax.ops.relaxation import soft_top_k

        x = jnp.zeros(3)
    else:
        torch = pytest.importorskip("torch")
        from omnibias.graph.torch.ops.relaxation import soft_top_k

        x = torch.zeros(3)
    with pytest.raises(ValueError, match="temperature"):
        soft_top_k(x, 1, temperature=temperature)
    # Preserve validation order when both arguments are invalid.
    with pytest.raises(ValueError, match="k must be"):
        soft_top_k(x, 0, temperature=temperature)


def test_selected_rows_backend_value_and_gradient_parity() -> None:
    """Distinct scores avoid comparing backend conventions at absolute-value cusps."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    torch = pytest.importorskip("torch")
    jax.config.update("jax_enable_x64", True)
    from omnibias.graph.jax.ops.relaxation import soft_top_k as jax_top_k
    from omnibias.graph.torch.ops.relaxation import soft_top_k as torch_top_k

    scores = SCORES[0]
    xj = jnp.asarray(scores, dtype=jnp.float64)
    xt = torch.tensor(scores, dtype=torch.float64, requires_grad=True)
    cotangent = np.asarray([0.2, -0.5, 1.3, 0.7, -0.2, 0.9])
    yj = jax_top_k(xj, 3, temperature=0.7)
    yt = torch_top_k(xt, 3, temperature=0.7)
    gj = jax.grad(lambda x: jnp.dot(jax_top_k(x, 3, temperature=0.7), cotangent))(xj)
    gt = torch.autograd.grad(yt, xt, grad_outputs=torch.tensor(cotangent))[0]
    np.testing.assert_allclose(np.asarray(yj), yt.detach().numpy(), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(np.asarray(gj), gt.numpy(), rtol=2e-13, atol=2e-13)
