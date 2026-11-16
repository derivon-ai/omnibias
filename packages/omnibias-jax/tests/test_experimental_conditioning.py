# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Conditioning of the damped Gauss-Newton step: diagnosis, the damping floor, and the plumbing into ``compile_residual``."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental import Residual, compile_residual  # noqa: E402
from omnibias.jax.experimental.conditioning import (  # noqa: E402
    ConditioningPolicy,
    condition_report,
    damped_solve,
    damping_floor,
    estimate_lambda_max,
    floored_damping,
    resolve_conditioning,
)

D = 4


def _psd(n, spread, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return jnp.asarray((Q * spread) @ Q.T), spread


def test_power_iteration_is_a_tight_lower_bound():
    spread = np.concatenate([[1e6], np.geomspace(1e-3, 1.0, 39)])
    K, _ = _psd(40, spread)
    lam = float(estimate_lambda_max(K, 16))
    assert lam <= 1e6 * (1 + 1e-9)
    assert lam >= 0.99e6  # a dominant mode converges fast


def test_policy_validation_and_resolution():
    assert resolve_conditioning(None).mode == "off"
    assert resolve_conditioning("floor").mode == "floor"
    p = ConditioningPolicy("floor", step_tolerance=1e-8)
    assert resolve_conditioning(p) is p
    with pytest.raises(ValueError):
        ConditioningPolicy("sometimes")
    with pytest.raises(ValueError):
        ConditioningPolicy(step_tolerance=0.0)


def test_report_and_floor_formulas():
    K, _ = _psd(30, np.concatenate([[1e8], np.geomspace(1e-2, 1.0, 29)]))
    pol = ConditioningPolicy("report", step_tolerance=1e-6)
    rep = condition_report(K, 1e-3, pol).to_config()
    eps = np.finfo(np.float64).eps
    assert rep["status"] == "conditioning_limited"
    assert rep["condition_estimate"] == pytest.approx((rep["lambda_max"] + 1e-3) / 1e-3)
    assert rep["expected_step_relative_error"] == pytest.approx(
        rep["condition_estimate"] * 64 * eps
    )
    c = 64 * eps
    assert rep["damping_floor"] == pytest.approx(rep["lambda_max"] * c / (1e-6 - c))
    assert not rep["floor_applied"] and rep["effective_damping"] == pytest.approx(1e-3)
    assert rep["lambda_max_upper"] >= rep["lambda_max"] * (
        1 - 1e-12
    )  # the rigorous bound is an upper bound
    # floor mode: the effective damping reaches the floor and the prediction meets the tolerance
    fl = condition_report(K, 1e-3, ConditioningPolicy("floor", step_tolerance=1e-6)).to_config()
    assert fl["floor_applied"] and fl["effective_damping"] == pytest.approx(fl["damping_floor"])
    assert fl["expected_step_relative_error"] <= 1e-6 * (1 + 1e-9)
    assert fl["status"] == "ok" and fl["requested_tolerance"] == 1e-6
    # a damping already above the floor is untouched
    big = condition_report(K, 1e3, ConditioningPolicy("floor")).to_config()
    assert not big["floor_applied"] and big["effective_damping"] == pytest.approx(1e3)
    # vector damping is floored elementwise
    d = floored_damping(K, jnp.asarray([1e-6, 1e-3, 1e5]), ConditioningPolicy("floor"))
    f = float(damping_floor(estimate_lambda_max(K, 16), K.dtype, ConditioningPolicy("floor")))
    np.testing.assert_allclose(d, [f, f, 1e5], rtol=1e-12)


def test_report_and_off_modes_change_nothing_bitwise():
    K, _ = _psd(25, np.geomspace(1e-4, 1e4, 25))
    r = jnp.asarray(np.random.default_rng(3).standard_normal(25))
    ref = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(K + 1e-3 * jnp.eye(25)), r)
    for pol in (None, ConditioningPolicy("off"), ConditioningPolicy("report")):
        alpha, d = damped_solve(K, r, 1e-3, pol)
        np.testing.assert_array_equal(alpha, ref)
        assert float(d) == 1e-3
    alpha, d = damped_solve(K, r, 1e-9, ConditioningPolicy("floor"))
    assert float(d) > 1e-9  # lambda_max ~ 1e4: the floor is about 1e4 * 64 eps / 1e-6 = 1.4e-4
    np.testing.assert_allclose(
        alpha,
        jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(K + float(d) * jnp.eye(25)), r),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------------------- compile_residual


def _init(widths, seed=0, out_scale=1.0):
    rng = np.random.default_rng(seed)
    layers = [
        (
            jnp.asarray(rng.standard_normal((a, b)) / np.sqrt(a)),
            jnp.asarray(0.2 * rng.standard_normal(b)),
        )
        for a, b in zip(widths[:-1], widths[1:], strict=True)
    ]
    W, b = layers[-1]
    layers[-1] = (out_scale * W, b)
    return layers


def mlp(params, x):
    a = x
    for W, b in params[:-1]:
        a = jnp.tanh(a @ W + b)
    W, b = params[-1]
    return (a @ W + b)[..., 0]


def _problem(out_scale=1.0, n=24, width=40):
    params = _init([D, width, width, 1], out_scale=out_scale)
    x = jnp.asarray(np.random.default_rng(1).standard_normal((n, D)))
    res = Residual(lambda q: q.lap[:, 0] + 0.3 * q.u[:, 0] - 0.1, ["value", "lap"])
    return params, [(x, res)]


def _flat(step):
    return np.concatenate([np.ravel(a) for a in jax.tree_util.tree_leaves(step)])


def test_default_reports_and_never_changes_the_step():
    params, blocks = _problem()
    rep_op = compile_residual(mlp, blocks, example_params=params, force="autodiff_rows")
    off_op = compile_residual(
        mlp, blocks, example_params=params, force="autodiff_rows", conditioning="off"
    )
    assert rep_op.explain()["conditioning"]["status"] == "no call yet"
    _, _, s_rep = rep_op(params, damping=1e-3)
    _, _, s_off = off_op(params, damping=1e-3)
    np.testing.assert_array_equal(_flat(s_rep), _flat(s_off))
    c = rep_op.explain()["conditioning"]
    assert c["mode"] == "report" and c["status"] in ("ok", "conditioning_limited", "unreliable")
    assert c["effective_damping"] == pytest.approx(1e-3) and not c["floor_applied"]
    assert c["lambda_max"] > 0 and c["condition_estimate"] >= 1
    assert off_op.explain()["conditioning"]["status"] == "not applicable"
    # the gram quantity has no solve: nothing to report
    gram = compile_residual(mlp, blocks, example_params=params, quantity="gram")
    gram(params)
    assert gram.explain()["conditioning"]["status"] == "not applicable"


def test_conditioning_limited_step_is_made_reproducible_by_the_floor():
    """The A31 pattern: engines agree on K to ~1e-14 yet differ in the damped step, because kappa(K + d I) is huge."""
    params, blocks = _problem(out_scale=3e3)
    damping = 1e-3
    steps = {}
    for mode in ("report", "floor"):
        for name in ("projected_streams", "autodiff_rows"):
            op = compile_residual(
                mlp,
                blocks,
                example_params=params,
                force=name,
                conditioning=ConditioningPolicy(mode, step_tolerance=1e-6),
            )
            _, K, step = op(params, damping=damping)
            steps[(mode, name)] = (_flat(step), op.explain()["conditioning"], np.asarray(K))
    K_a, K_b = steps[("report", "projected_streams")][2], steps[("report", "autodiff_rows")][2]
    assert np.max(np.abs(K_a - K_b)) / np.max(np.abs(K_a)) < 1e-11  # the Grams agree
    s_a, s_b = steps[("report", "projected_streams")][0], steps[("report", "autodiff_rows")][0]
    dev_report = np.max(np.abs(s_a - s_b)) / np.max(np.abs(s_a))
    rep = steps[("report", "autodiff_rows")][1]
    assert rep["status"] != "ok" and rep["expected_step_relative_error"] > 1e-6
    # the diagnosis is honest: the observed deviation does not exceed the prediction (up to a modest factor)
    assert dev_report <= 10 * rep["expected_step_relative_error"]
    f_a, f_b = steps[("floor", "projected_streams")][0], steps[("floor", "autodiff_rows")][0]
    dev_floor = np.max(np.abs(f_a - f_b)) / np.max(np.abs(f_a))
    fl = steps[("floor", "autodiff_rows")][1]
    assert fl["floor_applied"] and fl["effective_damping"] > damping
    assert fl["status"] == "ok" and fl["expected_step_relative_error"] <= 1e-6 * (1 + 1e-9)
    assert dev_floor <= 1e-6
    # The floor buys reproducibility: never worse than the unfloored deviation.  On a host whose BLAS makes the two
    # engines round identically both deviations sit at machine precision (observed 1.5578e-15 vs 1.5578e-15 on a farm
    # CPU), so a strict inequality would assert on the last bit; allow roundoff.
    assert dev_floor <= dev_report + 1e-14


def test_floor_is_consistent_across_engines_and_vector_damping():
    params, blocks = _problem(out_scale=50.0)
    pol = ConditioningPolicy("floor", step_tolerance=1e-8)
    outs = []
    for name in ("projected_streams", "full_streams", "autodiff_rows"):
        op = compile_residual(mlp, blocks, example_params=params, force=name, conditioning=pol)
        _, _, step = op(params, damping=jnp.asarray([1e-9, 1e-2]))
        outs.append(_flat(step))
    for o in outs[1:]:
        np.testing.assert_allclose(o, outs[0], rtol=1e-6, atol=1e-10)


def test_tolerance_is_never_asked_below_the_accuracy_of_the_matrix():
    K32 = jnp.asarray(np.diag([1.0, 1e-2, 1e-4]), dtype=jnp.float32)
    pol = ConditioningPolicy("floor", step_tolerance=1e-9)
    rep = condition_report(K32, 1e-6, pol).to_config()
    assert rep["step_tolerance"] > 1e-9 and rep["requested_tolerance"] == 1e-9
    assert np.isfinite(rep["damping_floor"]) and rep["expected_step_relative_error"] <= rep[
        "step_tolerance"
    ] * (1 + 1e-6)
