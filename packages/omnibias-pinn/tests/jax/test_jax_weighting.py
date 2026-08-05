# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the jax adaptive-weighting primitives.

Twin of ``tests/torch/test_torch_weighting.py``. Covers:

* :func:`grad_stats` / :func:`ntk_trace_stats` against hand-computed references
  over a parameter pytree.
* The measurement + shared state-machine loop equalising two terms' scales.
* :func:`reverse_gradient`: exact identity forward, exactly negated backward.
* :func:`self_adaptive_loss` and the :class:`SelfAdaptiveWeights` pytree:
  masked-mean value, the minimax under one ordinary descent update, mask
  selection, ``jax.jit`` / ``jax.grad`` compatibility, and pytree round-trip.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.pinn.jax.losses import (  # noqa: E402
    GradNormWeighter,
    estimate_ntk_trace,
    grad_stats,
    make_self_adaptive_weights,
    ntk_trace_stats,
    reverse_gradient,
    self_adaptive_loss,
)


@pytest.fixture
def params() -> dict[str, jax.Array]:
    return {
        "w": jnp.linspace(0.1, 0.8, 8, dtype=jnp.float64),
        "b": jnp.zeros(8, dtype=jnp.float64),
    }


@pytest.fixture
def x() -> jax.Array:
    return jnp.linspace(0.0, 1.0, 16, dtype=jnp.float64)


def _pde(p, x):
    return jnp.mean(jnp.sum(jnp.tanh(p["w"][None, :] * x[:, None] + p["b"]), 1) ** 2)


def _bc(p):
    return jnp.mean((jnp.tanh(p["b"]) - 1.0) ** 2)


# ---------------- grad_stats ----------------------------------------


def test_grad_stats_match_a_hand_computed_reference(params, x):
    fn = lambda p: _pde(p, x)  # noqa: E731
    got = grad_stats({"pde": fn}, params)
    grads = jax.tree_util.tree_leaves(jax.grad(fn)(params))
    flat = jnp.concatenate([jnp.abs(g).reshape(-1) for g in grads])
    assert got["pde"].max_abs == pytest.approx(float(jnp.max(flat)), rel=1e-14)
    assert got["pde"].mean_abs == pytest.approx(float(jnp.mean(flat)), rel=1e-14)


def test_untouched_leaves_count_as_zeros(params):
    """jax hands back an explicit zero, so the rule holds without extra work."""
    narrow = grad_stats({"bc": _bc}, params)["bc"]
    wide = grad_stats({"bc": _bc}, {**params, "unused": jnp.zeros(1000)})["bc"]
    assert wide.max_abs == pytest.approx(narrow.max_abs)
    assert wide.mean_abs < narrow.mean_abs


def test_grad_stats_are_non_negative_and_ordered(params, x):
    stats = grad_stats({"pde": lambda p: _pde(p, x), "bc": _bc}, params)
    for stat in stats.values():
        assert 0.0 <= stat.mean_abs <= stat.max_abs


def test_grad_stats_reject_empty(params):
    with pytest.raises(ValueError, match="empty"):
        grad_stats({}, params)


# ---------------- ntk_trace_stats -----------------------------------


def test_ntk_trace_stats_match_estimate_ntk_trace(params, x):
    fns = {"pde": lambda p: _pde(p, x), "bc": _bc}
    got = ntk_trace_stats(fns, params)
    for name, fn in fns.items():
        assert got[name] == pytest.approx(
            float(estimate_ntk_trace(fn, params)), rel=1e-12
        )


def test_ntk_trace_stats_reject_empty(params):
    with pytest.raises(ValueError, match="empty"):
        ntk_trace_stats({}, params)


# ---------------- measurement + state machine -----------------------


def test_gradnorm_loop_equalises_the_weighted_gradient_scales(params, x):
    fns = {"pde": lambda p: _pde(p, x), "bc": lambda p: 1e-4 * _bc(p)}
    stats = grad_stats(fns, params)
    assert stats["bc"].mean_abs < stats["pde"].mean_abs

    w = GradNormWeighter(["pde", "bc"], reference="pde", alpha=0.0)
    weights = w.update(stats)
    assert weights["bc"] > 1.0
    assert weights["bc"] * stats["bc"].mean_abs == pytest.approx(
        stats["pde"].max_abs, rel=1e-6
    )


def test_weighter_combine_stays_differentiable(params, x):
    w = GradNormWeighter(["pde", "bc"], reference="pde")
    w.update(grad_stats({"pde": lambda p: _pde(p, x), "bc": _bc}, params))
    g = jax.grad(lambda p: w.combine({"pde": _pde(p, x), "bc": _bc(p)}))(params)
    assert jnp.all(jnp.isfinite(g["w"]))


# ---------------- reverse_gradient ----------------------------------


def test_reverse_gradient_is_exactly_the_identity_forward():
    x = jnp.array([-3.25, 0.0, 1e-17, 7.5], dtype=jnp.float64)
    assert jnp.array_equal(reverse_gradient(x), x)


def test_reverse_gradient_negates_the_gradient_exactly():
    x = jnp.array([0.5, -2.0], dtype=jnp.float64)
    g = jax.grad(lambda v: jnp.sum(reverse_gradient(v) ** 2))(x)
    assert jnp.array_equal(g, -2.0 * x)


# ---------------- self-adaptive weights -----------------------------


def test_self_adaptive_loss_is_the_masked_mean():
    r = jnp.array([1.0, 2.0], dtype=jnp.float64)
    lam = jnp.zeros(2, dtype=jnp.float64)
    got = self_adaptive_loss(r, lam, mask="sigmoid", ascent=False)
    assert float(got) == pytest.approx(0.5 * (1.0 + 4.0) / 2)


@pytest.mark.parametrize(
    ("mask", "expected"), [("identity", 2.0), ("square", 4.0), ("relu", 2.0)]
)
def test_masks_apply_the_named_map(mask, expected):
    r = jnp.ones(1, dtype=jnp.float64)
    lam = jnp.full((1,), 2.0, dtype=jnp.float64)
    got = self_adaptive_loss(r, lam, mask=mask, ascent=False)
    assert float(got) == pytest.approx(expected)


def test_unknown_mask_is_rejected():
    with pytest.raises(ValueError, match="unknown mask"):
        self_adaptive_loss(jnp.ones(2), jnp.zeros(2), mask="not_an_activation")


def test_non_broadcastable_lambdas_rejected():
    with pytest.raises(ValueError, match="broadcast"):
        self_adaptive_loss(jnp.ones(3), jnp.ones(4))


def test_ascent_flips_only_the_weight_gradient(params, x):
    r = jnp.tanh(params["w"][None, :] * x[:, None]).sum(1)
    lam = jnp.zeros(r.shape[0], dtype=jnp.float64)
    g_up = jax.grad(lambda v: self_adaptive_loss(r, v, ascent=True))(lam)
    g_down = jax.grad(lambda v: self_adaptive_loss(r, v, ascent=False))(lam)
    assert jnp.array_equal(g_up, -g_down)
    assert float(self_adaptive_loss(r, lam, ascent=True)) == pytest.approx(
        float(self_adaptive_loss(r, lam, ascent=False))
    )


def test_one_descent_step_ascends_the_weights():
    """The minimax, driven by a single ordinary descent update."""
    saw = make_self_adaptive_weights(4)
    residual = jnp.array([0.01, 0.01, 1.0, 0.01], dtype=jnp.float64)
    for _ in range(50):
        g = jax.grad(lambda w: w.loss(residual))(saw)
        saw = saw.__class__(
            raw=saw.raw - 1.0 * g.raw, mask=saw.mask, ascent=saw.ascent
        )
    attention = saw.attention()
    assert int(jnp.argmax(attention)) == 2
    assert float(attention[2]) > float(attention[0])


def test_self_adaptive_weights_start_uniform_and_validate():
    saw = make_self_adaptive_weights(5)
    assert jnp.allclose(saw.attention(), jnp.full((5,), 0.5, dtype=jnp.float64))
    with pytest.raises(ValueError, match="n_points"):
        make_self_adaptive_weights(0)
    with pytest.raises(ValueError, match="weights"):
        saw.loss(jnp.ones(4))


def test_self_adaptive_weights_handle_vector_residuals():
    saw = make_self_adaptive_weights(3)
    got = saw.loss(jnp.ones((3, 2), dtype=jnp.float64))
    assert got.ndim == 0
    assert float(got) == pytest.approx(0.5)


def test_self_adaptive_weights_are_a_pytree():
    saw = make_self_adaptive_weights(6)
    leaves, treedef = jax.tree_util.tree_flatten(saw)
    assert len(leaves) == 1 and leaves[0].shape == (6,)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.array_equal(back.raw, saw.raw)
    assert back.mask == saw.mask and back.ascent == saw.ascent


def test_self_adaptive_loss_is_jittable():
    saw = make_self_adaptive_weights(4)
    r = jnp.linspace(-1.0, 1.0, 4, dtype=jnp.float64)
    jitted = jax.jit(lambda w, res: w.loss(res))
    assert float(jitted(saw, r)) == pytest.approx(float(saw.loss(r)))


def test_repr_reports_the_configuration():
    assert "n_points=3" in repr(make_self_adaptive_weights(3, mask="square"))
