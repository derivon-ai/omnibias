# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Error-free products, compensated Gram assembly and double-precision evaluation of float32 problems."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental import Residual, compile_operator, compile_residual  # noqa: E402
from omnibias.jax.experimental.compensated import (  # noqa: E402
    exact_gram,
    exact_matmul,
    slice_bits,
    split_slices,
)

D = 4


def _exact_product(A, B):
    """Correctly rounded float64 product by rational arithmetic (small matrices only)."""
    m, P = A.shape
    n = B.shape[1]
    out = np.empty((m, n))
    Af = [[Fraction(float(v)) for v in row] for row in A]
    Bf = [[Fraction(float(B[k, j])) for k in range(P)] for j in range(n)]
    for i in range(m):
        for j in range(n):
            out[i, j] = float(sum(a * b for a, b in zip(Af[i], Bf[j], strict=True)))
    return out


def test_slice_bits_keeps_the_sum_exact():
    for inner in (2, 10, 1000, 2**16, 2**20):
        beta = slice_bits(inner)
        assert inner * 2 ** (2 * beta) <= 2**53
    with pytest.raises(ValueError):
        slice_bits(2**50)
    with pytest.raises(ValueError):
        slice_bits(0)


def test_slices_sum_back_to_the_operand():
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.standard_normal((6, 40)) * 10.0 ** rng.integers(-6, 6, (6, 1)))
    pieces, rem = split_slices(A, axis=1, slices=4, beta=slice_bits(40))
    np.testing.assert_array_equal(sum(pieces) + rem, A)  # exact decomposition
    rel = jnp.max(jnp.abs(rem), axis=1) / jnp.max(jnp.abs(A), axis=1)
    assert float(jnp.max(rel)) < 2.0 ** -(4 * slice_bits(40) - 1)


def test_exact_matmul_is_correctly_rounded_where_the_plain_product_is_not():
    rng = np.random.default_rng(1)
    P = 300
    # terms of size 1e8 that cancel exactly along the inner axis, leaving an O(100) result
    big = rng.standard_normal((5, P // 2)) * 1e8
    A = np.concatenate([big, -big], axis=1) + rng.standard_normal((5, P))
    B = np.ones((P, 4)) * np.array([1.0, 2.0, 3.0, 4.0])
    ref = _exact_product(A, B)
    plain = np.asarray(jnp.asarray(A) @ jnp.asarray(B))
    comp = np.asarray(exact_matmul(jnp.asarray(A), jnp.asarray(B)))
    scale = np.abs(ref).max()
    err_plain = np.max(np.abs(plain - ref)) / scale
    err_comp = np.max(np.abs(comp - ref)) / scale
    assert err_comp <= 2 * np.finfo(float).eps
    assert err_plain > 1e6 * max(
        err_comp, np.finfo(float).eps
    )  # the plain product loses the answer to cancellation
    # two slices cannot hold both 1e8 and O(1) entries of one row to full accuracy: the dynamic range is the limit
    few = np.asarray(exact_matmul(jnp.asarray(A), jnp.asarray(B), slices=2))
    assert np.max(np.abs(few - ref)) / scale > 1e3 * np.finfo(float).eps


def test_exact_matmul_matches_the_plain_product_on_benign_data_and_validates_input():
    rng = np.random.default_rng(2)
    A, B = jnp.asarray(rng.standard_normal((7, 50))), jnp.asarray(rng.standard_normal((50, 9)))
    np.testing.assert_allclose(exact_matmul(A, B), A @ B, rtol=1e-13, atol=1e-13)
    with pytest.raises(TypeError):
        exact_matmul(A.astype(jnp.float32), B)
    with pytest.raises(ValueError):
        exact_matmul(A, B[:10])
    z = exact_matmul(jnp.zeros((3, 8)), jnp.ones((8, 2)))
    assert float(jnp.max(jnp.abs(z))) == 0.0


def test_exact_gram_is_exactly_symmetric_and_accurate():
    rng = np.random.default_rng(3)
    J = rng.standard_normal((8, 120)) * np.exp(rng.standard_normal((8, 1)) * 4)
    K = np.asarray(exact_gram(jnp.asarray(J)))
    np.testing.assert_array_equal(K, K.T)
    ref = _exact_product(J, J.T)
    assert (
        np.max(np.abs(K - ref) / np.sqrt(np.outer(np.diag(ref), np.diag(ref))))
        <= 4 * np.finfo(float).eps
    )


# --------------------------------------------------------------------------------- compile_residual


def _init(widths, seed=0):
    rng = np.random.default_rng(seed)
    return [
        (
            jnp.asarray(rng.standard_normal((a, b)) / np.sqrt(a)),
            jnp.asarray(0.2 * rng.standard_normal(b)),
        )
        for a, b in zip(widths[:-1], widths[1:], strict=True)
    ]


def mlp(params, x):
    a = x
    for W, b in params[:-1]:
        a = jnp.tanh(a @ W + b)
    W, b = params[-1]
    return (a @ W + b)[..., 0]


def _problem(n=10):
    params = _init([D, 16, 16, 1])
    x = jnp.asarray(np.random.default_rng(1).standard_normal((n, D)))
    return params, [(x, Residual(lambda q: q.lap[:, 0] + 0.3 * q.u[:, 0] - 0.1, ["value", "lap"]))]


def test_compensated_gram_agrees_with_the_standard_gram_and_reports_it():
    params, blocks = _problem()
    std = compile_residual(
        mlp, blocks, example_params=params, quantity="gram", force="autodiff_rows"
    )
    comp = compile_residual(
        mlp, blocks, example_params=params, quantity="gram", gram_precision="compensated"
    )
    r0, K0 = std(params)
    r1, K1 = comp(params)
    np.testing.assert_allclose(r1, r0, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(K1, K0, rtol=1e-11, atol=1e-11 * float(jnp.max(jnp.abs(K0))))
    ex = comp.explain()
    assert ex["gram_precision"] == "compensated" and ex["selected_strategy"] == "autodiff_rows"
    with pytest.raises(ValueError, match="cannot be combined"):
        compile_residual(
            mlp,
            blocks,
            example_params=params,
            gram_precision="compensated",
            force="projected_streams",
        )
    with pytest.raises(ValueError):
        compile_residual(mlp, blocks, example_params=params, gram_precision="exactish")


def test_compensated_step_matches_the_standard_step():
    params, blocks = _problem()
    std = compile_residual(mlp, blocks, example_params=params, force="autodiff_rows")
    comp = compile_residual(mlp, blocks, example_params=params, gram_precision="compensated")
    _, _, s0 = std(params, damping=1e-3)
    _, _, s1 = comp(params, damping=1e-3)
    a = np.concatenate([np.ravel(v) for v in jax.tree_util.tree_leaves(s0)])
    b = np.concatenate([np.ravel(v) for v in jax.tree_util.tree_leaves(s1)])
    np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-10 * np.abs(a).max())


# ----------------------------------------------------------------------------------- compute_dtype


def test_compute_dtype_float64_makes_a_float32_problem_accurate():
    params64 = _init([D, 24, 24, 1])
    x64 = jnp.asarray(np.random.default_rng(1).standard_normal((12, D)))
    to32 = lambda t: jax.tree_util.tree_map(lambda a: a.astype(jnp.float32), t)  # noqa: E731
    params32, x32 = to32(params64), x64.astype(jnp.float32)
    # the reference: float64 evaluation of the float32-rounded problem
    ref = compile_operator(mlp, (to32(params64), x32.astype(jnp.float64)))(
        jax.tree_util.tree_map(lambda a: a.astype(jnp.float64), params32), x32.astype(jnp.float64)
    )
    plain = compile_operator(mlp, (params32, x32))
    wide = compile_operator(mlp, (params32, x32), compute_dtype="float64")
    v_plain = plain(params32, x32)
    v_wide = wide(params32, x32)
    assert (
        v_plain.dtype == jnp.float32 and v_wide.dtype == jnp.float32
    )  # results come back in the input dtype
    e_plain = float(jnp.max(jnp.abs(v_plain - ref)) / jnp.max(jnp.abs(ref)))
    e_wide = float(jnp.max(jnp.abs(v_wide - ref)) / jnp.max(jnp.abs(ref)))
    assert e_wide <= 2 * np.finfo(np.float32).eps  # only the final rounding to float32
    assert e_wide < e_plain or e_plain <= 2 * np.finfo(np.float32).eps
    assert wide.explain()["compute_dtype"] == "float64"


def test_compute_dtype_validation():
    params = _init([D, 8, 1])
    x = jnp.asarray(np.random.default_rng(1).standard_normal((3, D)))
    with pytest.raises(ValueError):
        compile_operator(mlp, (params, x), compute_dtype="float16")
    params_, blocks = _problem(6)
    op = compile_residual(
        mlp, blocks, example_params=params_, quantity="gram", compute_dtype="float64"
    )
    r, K = op(params_)
    assert K.dtype == jnp.float64  # inputs were float64: nothing to narrow
