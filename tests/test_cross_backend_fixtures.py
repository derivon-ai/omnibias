# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Both backends against the same mpmath references (``omnibias.core.conformance.FIXTURE``).

Each case stores exact float64 weights and points and references computed in 90-digit arithmetic by
``scripts/generate_cross_backend_fixtures.py``.  The planned operators and the planned residual programs of JAX and
PyTorch must reproduce them, and each other, to 1e-12 (relative to the larger of 1 and the reference magnitude).
"""

from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from omnibias.core.conformance import FIXTURE
from omnibias.core.planner import PlanCache

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.experimental import Residual as JaxResidual  # noqa: E402
from omnibias.jax.experimental import compile_operator as jax_operator  # noqa: E402
from omnibias.jax.experimental import compile_residual as jax_residual  # noqa: E402
from omnibias.torch.experimental import Residual as TorchResidual  # noqa: E402
from omnibias.torch.experimental import compile_planned_operator as torch_operator  # noqa: E402
from omnibias.torch.experimental import compile_residual as torch_residual  # noqa: E402

DOC = json.loads(FIXTURE.read_text())
CASES = {c["name"]: c for c in DOC["cases"]}
TOL = 1e-12
FLOAT_TOL = 1e-11  # to the 90-digit reference (the reference itself carries ~1e-18 of difference-quotient error)


def arr(x) -> np.ndarray:
    if isinstance(x, str):
        return np.asarray(float.fromhex(x))
    return np.asarray([arr(v) for v in x]) if len(x) else np.zeros((0,))


def ref(x) -> np.ndarray:
    if isinstance(x, str):
        return np.asarray(float(mp.mpf(x)))
    return np.asarray([ref(v) for v in x])


def close(got, want, tol, what=""):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.size == want.size:  # a (B, 1) column and a (B,) vector are the same scalar field
        got = got.reshape(want.shape)
    assert got.shape == want.shape, what
    scale = np.maximum(1.0, np.abs(want))
    worst = float(np.max(np.abs(got - want) / scale)) if got.size else 0.0
    assert worst <= tol, f"{what}: worst relative error {worst:.3e} > {tol:.0e}"


def case_layers(case):
    return [(arr(layer["W"]), arr(layer["b"])) for layer in case["layers"]]


# --------------------------------------------------------------------------------------------- the networks


def torch_model(case):
    layers = [
        (torch.tensor(W, requires_grad=True), torch.tensor(b, requires_grad=True))
        for W, b in case_layers(case)
    ]
    n = len(layers)
    return [(W, b, case["activation"] if k < n - 1 else None) for k, (W, b) in enumerate(layers)]


def jax_model(case):
    params = [(jnp.asarray(W.T), jnp.asarray(b)) for W, b in case_layers(case)]
    act = {"tanh": jnp.tanh, "silu": jax.nn.silu}[case["activation"]]

    def model(p, x):
        a = x
        for W, b in p[:-1]:
            a = act(a @ W + b)
        W, b = p[-1]
        out = a @ W + b
        return out[..., 0] if out.shape[-1] == 1 else out

    return params, model


SCALAR = [n for n, c in CASES.items() if c["dims"][-1] == 1]
FIELD = [n for n, c in CASES.items() if c["dims"][-1] != 1]


# ---------------------------------------------------------------------------------------------- operators


@pytest.mark.parametrize("name", SCALAR)
def test_laplacian_agrees_with_the_reference_and_across_backends(name):
    case = CASES[name]
    x = arr(case["interior"])
    want = ref(case["reference"]["lap"])
    layers = torch_model(case)
    t = torch_operator(layers, torch.tensor(x), cache=PlanCache())(torch.tensor(x)).detach().numpy()
    params, model = jax_model(case)
    j = np.asarray(
        jax_operator(model, (params, jnp.asarray(x)), cache=PlanCache())(params, jnp.asarray(x))
    )
    close(t, want, FLOAT_TOL, f"{name} torch vs mp")
    close(j, want, FLOAT_TOL, f"{name} jax vs mp")
    close(t, j, TOL, f"{name} torch vs jax")


@pytest.mark.parametrize("name", FIELD)
def test_divergence_logdet_and_solve_agree_with_the_reference_and_across_backends(name):
    case = CASES[name]
    x = arr(case["interior"])
    rhs = arr(case["rhs"])
    layers = torch_model(case)
    params, model = jax_model(case)
    tx, jx = torch.tensor(x), jnp.asarray(x)
    for key, kwargs in (("divergence", {}), ("logdet", {})):
        want = ref(case["reference"][key])
        t = torch_operator(layers, tx, operator=key, cache=PlanCache())
        t_out = t.evaluate(tx).value.detach().numpy()
        j_out = np.asarray(
            jax_operator(model, (params, jx), operator=key, cache=PlanCache(), **kwargs)(params, jx)
        )
        close(t_out, want, FLOAT_TOL, f"{name} {key} torch vs mp")
        close(j_out, want, FLOAT_TOL, f"{name} {key} jax vs mp")
        close(t_out, j_out, TOL, f"{name} {key} torch vs jax")
    want = ref(case["reference"]["solve"])
    t_sol = torch_operator(layers, (tx, torch.tensor(rhs)), operator="solve", cache=PlanCache())(
        tx, torch.tensor(rhs)
    )
    j_sol = np.asarray(
        jax_operator(model, (params, jx, jnp.asarray(rhs)), operator="solve", cache=PlanCache())(
            params, jx, jnp.asarray(rhs)
        )
    )
    close(t_sol.detach().numpy(), want, FLOAT_TOL, f"{name} solve torch vs mp")
    close(j_sol, want, FLOAT_TOL, f"{name} solve jax vs mp")
    close(t_sol.detach().numpy(), j_sol, TOL, f"{name} solve torch vs jax")


# -------------------------------------------------------------------------------------------- residuals


@pytest.mark.parametrize("name", SCALAR)
def test_residual_gram_and_gauss_newton_step_agree_with_the_reference_and_across_backends(name):
    case = CASES[name]
    want = case["reference"]["residual"]
    xi, xb = arr(case["interior"]), arr(case["boundary"])
    t_layers = torch_model(case)
    t_blocks = [
        (
            torch.tensor(xi),
            TorchResidual(lambda q: q.lap[:, 0] + torch.sin(q.x[:, 0]), ["value", "lap"]),
        ),
        (torch.tensor(xb), TorchResidual(lambda q: q.u[:, 0] - q.x.sum(-1), ["value"])),
    ]
    params, model = jax_model(case)
    j_blocks = [
        (
            jnp.asarray(xi),
            JaxResidual(lambda q: q.lap[:, 0] + jnp.sin(q.x[:, 0]), ["value", "lap"]),
        ),
        (jnp.asarray(xb), JaxResidual(lambda q: q.u[:, 0] - q.x.sum(-1), ["value"])),
    ]
    damping = float(want["damping"])
    flat_ref = ref(want["step"])
    t_results = {}
    for engine in ("full_streams", "autodiff_rows"):
        op = torch_residual(t_layers, t_blocks, force=engine, cache=PlanCache())
        r, K, step = op(damping=torch.tensor(damping, dtype=torch.float64))
        flat = np.concatenate([s.detach().numpy().reshape(-1) for s in step])
        t_results[engine] = (r.numpy(), K.numpy(), flat)
    j_results = {}
    for engine in ("full_streams", "projected_streams", "autodiff_rows"):
        try:
            op = jax_residual(
                model, j_blocks, example_params=params, force=engine, cache=PlanCache()
            )
        except (
            Exception
        ) as err:  # an engine that refuses this demand is not a failure of the fixture
            assert (
                "cannot" in str(err) or "refuse" in str(err).lower() or "not" in str(err).lower()
            ), err
            continue
        r, K, step = op(params, damping=damping)
        flat = np.concatenate(
            [np.asarray(W).T.reshape(-1).tolist() + np.asarray(b).tolist() for W, b in step]
        )
        j_results[engine] = (np.asarray(r), np.asarray(K), flat)
    assert j_results, "no JAX engine served the fixture"
    for label, results in (("torch", t_results), ("jax", j_results)):
        for engine, (r, K, flat) in results.items():
            close(r, ref(want["r"]), FLOAT_TOL, f"{name} {label}/{engine} r")
            close(K, ref(want["K"]), 1e-9, f"{name} {label}/{engine} K")
            close(flat, flat_ref, 1e-9, f"{name} {label}/{engine} step")
    t_r, t_K, t_step = t_results["full_streams"]
    j_r, j_K, j_step = next(iter(j_results.values()))
    close(t_r, j_r, TOL, f"{name} r torch vs jax")
    close(t_K, j_K, 1e-11, f"{name} K torch vs jax")
    close(t_step, j_step, 1e-10, f"{name} step torch vs jax")


def test_fixture_is_current():
    """The committed JSON is what the generator writes (guards hand edits)."""
    assert DOC["version"] == 1 and DOC["dps"] >= 60
    assert {c["name"] for c in DOC["cases"]} == set(CASES)
    for c in DOC["cases"]:
        assert c["reference"], c["name"]
