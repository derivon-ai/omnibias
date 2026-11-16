# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Generate the operator coverage matrix of one backend and render ``docs/operator-coverage.md``.

Every cell builds a small problem, asks the planner to serve it, evaluates the planned program and compares the value
with a dense autodiff oracle written in this file (nested ``hessian`` traces of the model's own forward pass, Jacobians
of the residual vector by ``jacfwd``), so no derivative formula of the package is involved.  A cell records

* ``structured``   a chain program (projected, forward Laplacian, polylaplacian, Woodbury, streams) served it,
* ``reduced``      a generic exact engine served a model that is not a dense chain, along only ``min(rows, columns)``
                   directions of its first-layer products (the subspace programs),
* ``fallback``     a generic exact engine served it (nested autodiff or Taylor-mode jets) without a reduction,
* ``composed``     it is built from calls of structured programs,
* ``refused``      the planner refused it and said why,
* ``not_provided`` the package has no program for it (the reason names the route that exists),

and the largest relative error against the oracle.  A cell that disagrees with its oracle is a ``mismatch`` and the
script exits non-zero.

usage: python scripts/generate_coverage_matrix.py run jax|torch [--out docs/coverage/matrix_<backend>.json] [--only SECTION]
       python scripts/generate_coverage_matrix.py render
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "docs" / "coverage"
PAGE = ROOT / "docs" / "operator-coverage.md"
TOL = 1e-8
STATUSES = (
    "structured",
    "reduced",
    "fallback",
    "composed",
    "refused",
    "not_provided",
    "mismatch",
    "error",
)
STRUCTURED_ALGORITHMS = frozenset(
    {
        "projected",
        "forward_laplacian",
        "forward_polylaplacian",
        "woodbury",
        "projected_streams",
        "full_streams",
    }
)
D = 12  # input dimension of the architecture rows (more than the width of any first-layer product)


# --------------------------------------------------------------------------------------------------- cell plumbing


def _array(a: Any) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=float)


def rel(a: Any, b: Any) -> float:
    a, b = _array(a), _array(b)
    if a.shape != b.shape:
        a = a.reshape(b.shape)
    return float(np.max(np.abs(a - b)) / (1.0 + np.max(np.abs(b))))


def cell(
    status: str,
    algorithm: str | None = None,
    *,
    error: float | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    assert status in STATUSES, status
    out: dict[str, Any] = {"status": status}
    if algorithm is not None:
        out["algorithm"] = algorithm
    if error is not None:
        out["error"] = error
    if note:
        out["note"] = note
    return out


def checked(
    status: str, algorithm: str, got: Any, want: Any, *, tol: float = TOL, note: str | None = None
) -> dict[str, Any]:
    err = rel(got, want)
    if not np.isfinite(err) or err > tol:
        return cell("mismatch", algorithm, error=err, note=note)
    return cell(status, algorithm, error=err, note=note)


def guarded(fn: Any) -> dict[str, Any]:
    """Run a cell; a ``ValueError`` from the planner is a refusal with its reason, anything else is an error."""
    try:
        return fn()
    except ValueError as err:
        return cell("refused", note=str(err).replace("\n", " ")[:300])
    except Exception as err:  # noqa: BLE001 - recorded, and the gate fails on it
        return cell("error", note=f"{type(err).__name__}: {str(err)[:300]}")


# ---------------------------------------------------------------------------------------------------------- JAX


def run_jax(only: set[str] | None) -> dict[str, Any]:
    # The matrix pins the built-in program pool: an installed ``folx`` would otherwise be a planner candidate and change the
    # algorithm named in generic cells with the host's packages (the status and the oracle agreement do not change).
    os.environ["OMNIBIAS_FOLX_ENGINE"] = "off"
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from omnibias.core.planner import PlanCache
    from omnibias.jax.activations import get_activation, list_activations
    from omnibias.jax.experimental import Residual, compile_operator, compile_residual

    rng = np.random.default_rng(0)

    def mk(a: int, b: int) -> tuple[Any, Any]:
        return jnp.asarray(rng.standard_normal((a, b)) / np.sqrt(a)), jnp.asarray(
            0.1 * rng.standard_normal(b)
        )

    # ------------------------------------------------------------------------------------------- architectures
    H, T, DT = 5, 3, 2  # width, attention tokens, token width
    Dm = D

    def mlp(params, x):
        a = x
        for W, b in params[:-1]:
            a = jnp.tanh(a @ W + b)
        W, b = params[-1]
        return (a @ W + b)[..., 0]

    def siren(params, x):
        a = x
        for k, (W, b) in enumerate(params[:-1]):
            z = a @ W + b
            a = jnp.sin(7.0 * z) if k == 0 else jnp.sin(z)
        W, b = params[-1]
        return (a @ W + b)[..., 0]

    def residual_mlp(params, x):
        (W1, b1), *blocks, (Wo, bo) = params
        h = jnp.tanh(x @ W1 + b1)
        for W, b in blocks:
            h = h + jnp.tanh(h @ W + b)
        return (h @ Wo + bo)[..., 0]

    def fourier_concat(params, x):
        B, (W1, b1), (Wo, bo) = params
        phi = jnp.concatenate([jnp.sin(x @ B), jnp.cos(x @ B)], axis=-1)
        h = jnp.tanh(phi @ W1 + b1)
        return (h @ Wo + bo)[..., 0]

    def gated(params, x):
        B, (Wu, bu), (Wv, bv), (W1, b1), (W2, b2), (Wo, bo) = params
        phi = jnp.concatenate([jnp.sin(x @ B), jnp.cos(x @ B)], axis=-1)
        u, v = jnp.tanh(phi @ Wu + bu), jnp.tanh(phi @ Wv + bv)
        h = phi
        for W, b in ((W1, b1), (W2, b2)):
            f = jnp.tanh(h @ W + b)
            h = f * u + (1.0 - f) * v
        return (h @ Wo + bo)[..., 0]

    def attention(params, x):
        Win, Wq, Wk, Wv, Wo = params
        z = (x @ Win).reshape(*x.shape[:-1], T, DT)
        q, k, v = z @ Wq, z @ Wk, z @ Wv
        a = jax.nn.softmax(q @ jnp.swapaxes(k, -1, -2) / np.sqrt(DT), axis=-1)
        o = (a @ v).reshape(*x.shape[:-1], T * DT)
        return (jnp.tanh(o) @ Wo)[..., 0]

    def kan(params, x):
        c, (Wo, bo) = params
        freqs = jnp.arange(1, c.shape[-1] + 1, dtype=x.dtype)
        basis = jnp.sin(x[..., None] * freqs)  # (B, D, K): every edge reads one coordinate
        h = jnp.tanh(jnp.einsum("...ik,ijk->...j", basis, c))
        return (h @ Wo + bo)[..., 0]

    def hard_bc(params, x):
        return x[..., 0] * (1.0 - x[..., 0]) * mlp(params, x)

    archs: dict[str, tuple[Any, Any]] = {
        "mlp_tanh": (mlp, [mk(Dm, H), mk(H, H), mk(H, 1)]),
        "siren_scaled": (siren, [mk(Dm, H), mk(H, H), mk(H, 1)]),
        "residual_mlp": (residual_mlp, [mk(Dm, H), mk(H, H), mk(H, H), mk(H, 1)]),
        "fourier_features": (
            fourier_concat,
            (jnp.asarray(rng.standard_normal((Dm, 4))), mk(8, H), mk(H, 1)),
        ),
        "gated_pirate": (
            gated,
            (
                jnp.asarray(rng.standard_normal((Dm, 4))),
                mk(8, H),
                mk(8, H),
                mk(8, H),
                mk(H, H),
                mk(H, 1),
            ),
        ),
        "attention": (
            attention,
            (
                jnp.asarray(rng.standard_normal((Dm, T * DT)) / np.sqrt(Dm)),
                jnp.asarray(rng.standard_normal((DT, DT))),
                jnp.asarray(rng.standard_normal((DT, DT))),
                jnp.asarray(rng.standard_normal((DT, DT))),
                jnp.asarray(rng.standard_normal((T * DT, 1)) / np.sqrt(T * DT)),
            ),
        ),
        "kan_edges": (kan, (jnp.asarray(rng.standard_normal((Dm, H, 3)) / np.sqrt(Dm)), mk(H, 1))),
        "hard_boundary_mask": (hard_bc, [mk(Dm, H), mk(H, H), mk(H, 1)]),
    }
    x = jnp.asarray(rng.standard_normal((4, Dm)) * 0.8)
    S_full = jnp.asarray(rng.standard_normal((Dm, Dm)) / np.sqrt(Dm))

    def lap_ref(fn, params, xs, power=1, A=None):
        def point(z):
            return fn(params, z[None])[0]

        g = point
        for _ in range(power):

            def lap(z, g=g):
                Hz = jax.hessian(g)(z)
                return jnp.trace(Hz if A is None else A @ Hz)

            g = lap
        return jax.vmap(g)(xs)

    def classify(op: Any) -> str:
        if op.matched:
            return "structured" if op.algorithm in STRUCTURED_ALGORITHMS else "fallback"
        return "reduced" if op.algorithm.startswith("subspace") else "fallback"

    sections: dict[str, Any] = {}
    # ``--only architectures:NAME[,architectures:NAME2]`` runs single rows (the reference of a high power on a large model
    # compiles a lot; a row per job keeps the host's memory bounded); the row files merge by union.
    arch_rows = {o.split(":", 1)[1] for o in (only or ()) if o.startswith("architectures:")}

    def want(name: str) -> bool:
        return only is None or name in only or (name == "architectures" and bool(arch_rows))

    if want("architectures"):
        ops = {
            "laplacian": {"power": 1},
            "weighted_laplacian": {"power": 1, "diffusion": S_full},
            "biharmonic": {"power": 2},
            "triharmonic": {"power": 3},
        }
        table: dict[str, dict[str, Any]] = {}
        for aname, (fn, params) in archs.items():
            if arch_rows and aname not in arch_rows:
                continue
            row: dict[str, Any] = {}
            for oname, kw in ops.items():

                def run(fn=fn, params=params, kw=kw):
                    xs = x[:2] if kw["power"] == 3 else x
                    op = compile_operator(
                        fn, (params, xs), operator="laplacian", cache=PlanCache(), **kw
                    )
                    A = None if "diffusion" not in kw else S_full @ S_full.T
                    return checked(
                        classify(op),
                        op.algorithm,
                        op(params, xs),
                        lap_ref(fn, params, xs, kw["power"], A),
                    )

                row[oname] = guarded(run)
                jax.clear_caches()  # drop the compiled programs of the cell: the references are large traces
            table[aname] = row
        sections["architectures"] = table

    # ---------------------------------------------------------------------------------------------- activations
    def chain_layers(name: str) -> list[tuple[Any, Any, Any]]:
        g = np.random.default_rng(5)
        W1 = jnp.asarray(g.standard_normal((6, 4)) * 0.5)
        W2 = jnp.asarray(g.standard_normal((6, 6)) * 0.4)
        Wo = jnp.asarray(g.standard_normal((1, 6)) * 0.5)
        bs = [jnp.asarray(0.3 * g.standard_normal(n)) for n in (6, 6, 1)]
        return [(W1, bs[0], name), (W2, bs[1], name), (Wo, bs[2], None)]

    if want("activations"):
        xa = jnp.asarray(np.random.default_rng(6).standard_normal((3, 4)) * 0.6)
        table = {}
        for name in list_activations():
            spec = get_activation(name)
            row = {}
            if spec.fastpath is None:
                table[name] = {
                    "_note": cell("not_provided", note="no closed-form derivative kernel")
                }
                continue
            layers = chain_layers(name)

            def fwd(layers, z):
                a = z
                for W, b, act in layers:
                    a = a @ W.T + b
                    if act is not None:
                        a = get_activation(act).forward(a)
                return a[..., 0]

            def oracle(power, xs=xa, layers=layers):
                def point(z):
                    return fwd(layers, z[None])[0]

                g = point
                for _ in range(power):
                    g = (lambda g: lambda z: jnp.trace(jax.hessian(g)(z)))(g)
                return jax.vmap(g)(xa)

            for oname, kw in (
                ("laplacian", {"power": 1}),
                ("laplacian_projected", {"power": 1, "force": "projected"}),
                ("laplacian_forward", {"power": 1, "force": "forward_laplacian"}),
                ("biharmonic", {"power": 2}),
            ):

                def run(layers=layers, kw=kw, oracle=oracle):
                    try:
                        op = compile_operator(
                            layers, xa, operator="laplacian", cache=PlanCache(), **kw
                        )
                    except ValueError as err:
                        op = compile_operator(
                            layers,
                            xa,
                            operator="laplacian",
                            cache=PlanCache(),
                            allow_almost_everywhere=True,
                            **kw,
                        )
                        res = checked(
                            classify(op), op.algorithm, op(layers, xa), oracle(kw["power"])
                        )
                        res["status"] = "refused" if res["status"] != "mismatch" else "mismatch"
                        res["note"] = (
                            "opt-in: allow_almost_everywhere; " + str(err).replace("\n", " ")[:200]
                        )
                        return res
                    return checked(classify(op), op.algorithm, op(layers, xa), oracle(kw["power"]))

                row[oname] = guarded(run)

            def run_gram(layers=layers, name=name):
                params = [(W, b) for W, b, _ in layers]
                blk = (xa, Residual(lambda q: q.lap[:, 0] + 0.5 * q.u[:, 0], ["value", "lap"]))
                op = compile_residual(layers, [blk], quantity="gram", cache=PlanCache())

                def r_of(p):
                    ls = [(W, b, a) for (W, b), (_, _, a) in zip(p, layers, strict=True)]

                    def point(z):
                        return fwd(ls, z[None])[0]

                    def one(z):
                        return jnp.trace(jax.hessian(point)(z)) + 0.5 * point(z)

                    return jax.vmap(one)(xa)

                flat, unravel = ravel_pytree(params)
                J = jax.jacfwd(lambda f: r_of(unravel(f)))(flat)
                _, K = op(layers)
                status = "structured" if op.algorithm in STRUCTURED_ALGORITHMS else "fallback"
                return checked(status, op.algorithm, K, J @ J.T, tol=1e-7)

            row["residual_gram"] = guarded(run_gram)
            table[name] = row
        sections["activations"] = table

    # ------------------------------------------------------------------------------------------- vector fields
    if want("vector_fields"):
        g = np.random.default_rng(7)
        Dv = 6

        def mkv(a, b):
            return jnp.asarray(g.standard_normal((a, b)) / np.sqrt(a)), jnp.asarray(
                0.1 * g.standard_normal(b)
            )

        vec_models: dict[str, tuple[Any, Any]] = {}

        def mlp_vec(params, x):
            a = x
            for W, b in params[:-1]:
                a = jnp.tanh(a @ W + b)
            W, b = params[-1]
            return a @ W + b

        def residual_vec(params, x):
            (W1, b1), (W2, b2), (Wo, bo) = params
            h = jnp.tanh(x @ W1 + b1)
            h = h + jnp.tanh(h @ W2 + b2)
            return h @ Wo + bo

        vec_models = {
            "mlp_tanh": (mlp_vec, [mkv(Dv, 8), mkv(8, 8), mkv(8, Dv)]),
            "residual_mlp": (residual_vec, [mkv(Dv, 4), mkv(4, 4), mkv(4, Dv)]),
        }
        xv = jnp.asarray(g.standard_normal((3, Dv)))
        rhs = jnp.asarray(g.standard_normal((3, Dv)))
        table = {}
        for aname, (fn, params) in vec_models.items():
            row = {}

            def jac(z, fn=fn, params=params):
                return jax.jacfwd(lambda w: fn(params, w[None])[0])(z)

            def run_div(fn=fn, params=params):
                op = compile_operator(fn, (params, xv), operator="divergence", cache=PlanCache())
                ref = jax.vmap(lambda z: jnp.trace(jac(z, fn, params)))(xv)
                return checked(classify(op), op.algorithm, op(params, xv), ref)

            def run_logdet(fn=fn, params=params):
                op = compile_operator(fn, (params, xv), operator="logdet", cache=PlanCache())
                ref = jax.vmap(lambda z: jnp.linalg.slogdet(jnp.eye(Dv) + jac(z, fn, params))[1])(
                    xv
                )
                return checked(classify(op), op.algorithm, op(params, xv), ref)

            def run_solve(fn=fn, params=params):
                op = compile_operator(fn, (params, xv), operator="solve", cache=PlanCache())
                ref = jax.vmap(lambda z, r: jnp.linalg.solve(jnp.eye(Dv) + jac(z, fn, params), r))(
                    xv, rhs
                )
                return checked(classify(op), op.algorithm, op(params, xv, rhs), ref)

            def run_vlap(fn=fn, params=params):
                op = compile_operator(fn, (params, xv), operator="laplacian", cache=PlanCache())
                ref = jax.vmap(
                    lambda z: jnp.trace(
                        jax.jacfwd(jax.jacfwd(lambda w: fn(params, w[None])[0]))(z),
                        axis1=-2,
                        axis2=-1,
                    )
                )(xv)
                return checked(classify(op), op.algorithm, op(params, xv), ref)

            row["divergence"] = guarded(run_div)
            row["logdet"] = guarded(run_logdet)
            row["solve"] = guarded(run_solve)
            row["vector_laplacian"] = guarded(run_vlap)
            row["curl"] = cell(
                "not_provided",
                note="first-order: jax.jacfwd(model) gives the antisymmetric part in D forward passes",
            )
            table[aname] = row
        sections["vector_fields"] = table

    # ------------------------------------------------------------------------------- derivative combinations
    if want("derivatives"):
        params = [mk(5, 8), mk(8, 8), mk(8, 1)]
        xd = jnp.asarray(np.random.default_rng(9).standard_normal((4, 5)) * 0.8)
        layers = [(W.T, b, "tanh" if k < 2 else None) for k, (W, b) in enumerate(params)]
        table = {}

        def hess(z):
            return jax.hessian(lambda w: mlp(params, w[None])[0])(z)

        def run_mixed():
            i, j = 1, 3
            plus = jnp.zeros((5, 1)).at[i, 0].set(0.5).at[j, 0].set(0.5)
            minus = jnp.zeros((5, 1)).at[i, 0].set(0.5).at[j, 0].set(-0.5)
            a = compile_operator(
                layers, xd, operator="laplacian", diffusion=plus, cache=PlanCache()
            )
            b = compile_operator(
                layers, xd, operator="laplacian", diffusion=minus, cache=PlanCache()
            )
            ref = jax.vmap(lambda z: hess(z)[i, j])(xd)
            return checked(
                "composed", f"{a.algorithm}+{b.algorithm}", a(layers, xd) - b(layers, xd), ref
            )

        def run_time():
            e = np.zeros((5, 1))
            e[0, 0] = 1.0
            op = compile_operator(layers, xd, operator="laplacian", directions=e, cache=PlanCache())
            out = op.evaluate(layers, xd)
            ref = jax.vmap(lambda z: jax.grad(lambda w: mlp(params, w[None])[0])(z)[0])(xd)
            return checked(classify(op), op.algorithm, out.aux["directional"][..., 0], ref)

        def run_subset():
            coords = [1, 2, 3]
            op = compile_operator(
                layers, xd, operator="laplacian", coordinates=coords, cache=PlanCache()
            )
            ref = jax.vmap(lambda z: jnp.trace(hess(z)[1:4, 1:4]))(xd)
            return checked(classify(op), op.algorithm, op(layers, xd), ref)

        table["mixed_derivative"] = guarded(run_mixed)
        table["time_derivative"] = guarded(run_time)
        table["space_subset_laplacian"] = guarded(run_subset)
        table["curl"] = cell("not_provided", note="see vector_fields")
        sections["derivatives"] = table

    # ------------------------------------------------------------------------------------------------ residuals
    if want("residuals"):
        g = np.random.default_rng(11)
        Dr = 3
        params = [mk(Dr, 8), mk(8, 8), mk(8, 1)]
        layers = [(W.T, b, "tanh" if k < 2 else None) for k, (W, b) in enumerate(params)]
        xr = jnp.asarray(g.standard_normal((5, Dr)) * 0.8)
        a_coeff = jnp.asarray(1.0 + 0.5 * g.random(5))
        f_src = jnp.asarray(g.standard_normal(5))
        normal = jnp.asarray(g.standard_normal((5, Dr)))
        normal = normal / jnp.linalg.norm(normal, axis=1, keepdims=True)
        table = {}

        def point_u(p, z):
            a = z
            for k, (W, b) in enumerate(p):
                a = a @ W + b
                if k < 2:
                    a = jnp.tanh(a)
            return a[0]

        def quantities(p, z, coords=None):
            u = point_u(p, z)
            gr = jax.grad(lambda w: point_u(p, w))(z)
            Hz = jax.hessian(lambda w: point_u(p, w))(z)
            lap = jnp.trace(Hz if coords is None else Hz[jnp.ix_(coords, coords)])
            bil = jnp.trace(
                jax.hessian(lambda w: jnp.trace(jax.hessian(lambda v: point_u(p, v))(w)))(z)
            )
            return u, gr, lap, bil

        def residual_cell(
            name, fn, demand, rows, *, coordinates=None, order=None, projected_ok=True
        ):
            def run():
                block = (
                    (xr, Residual(fn, demand))
                    if order is None
                    else (xr, Residual(fn, demand), order)
                )
                op = compile_residual(
                    layers, [block], quantity="gram", coordinates=coordinates, cache=PlanCache()
                )
                coords = None if coordinates is None else jnp.asarray(coordinates)
                flat, unravel = ravel_pytree(params)

                def r_of(f):
                    p = unravel(f)
                    outs = jax.vmap(lambda z: jnp.stack(quantities_flat(p, z, coords)))(xr)
                    return rows(outs, xr)

                def quantities_flat(p, z, coords):
                    u, gr, lap, bil = quantities(p, z, coords)
                    return [u, lap, bil, *gr]

                J = jax.jacfwd(r_of)(flat)
                _, K = op(layers)
                status = "structured" if op.algorithm in STRUCTURED_ALGORITHMS else "fallback"
                return checked(status, op.algorithm, K, J @ J.T, tol=1e-7)

            table[name] = guarded(run)

        residual_cell(
            "variable_coefficient_poisson",
            lambda q: a_coeff * q.lap[:, 0] - f_src,
            ["value", "lap"],
            lambda o, x_: a_coeff * o[:, 1] - f_src,
        )
        residual_cell(
            "allen_cahn_nonlinear",
            lambda q: 0.01 * q.lap[:, 0] + q.u[:, 0] - q.u[:, 0] ** 3,
            ["value", "lap"],
            lambda o, x_: 0.01 * o[:, 1] + o[:, 0] - o[:, 0] ** 3,
        )
        residual_cell(
            "burgers_time_space",
            lambda q: q.grad[:, 0, 0] + q.u[:, 0] * q.grad[:, 1, 0] - 0.05 * q.lap[:, 0],
            ["value", "grad", "lap"],
            lambda o, x_: o[:, 3] + o[:, 0] * o[:, 4] - 0.05 * o[:, 1],
            coordinates=[1],
        )
        residual_cell(
            "boundary_normal_derivative",
            lambda q: jnp.sum(q.grad[:, :, 0] * normal, axis=1) - f_src,
            ["value", "grad"],
            lambda o, x_: jnp.sum(o[:, 3:6] * normal, axis=1) - f_src,
            order=1,
        )
        residual_cell(
            "variable_coefficient_biharmonic",
            lambda q: a_coeff * q.bilap[:, 0] - f_src,
            ["value", "grad_sq", "lap", "bilap"],
            lambda o, x_: a_coeff * o[:, 2] - f_src,
            order=4,
        )
        residual_cell(
            "gradient_norm_nonlinearity",
            lambda q: q.lap[:, 0] + q.grad_sq[:, 0] - f_src,
            ["value", "grad_sq", "lap"],
            lambda o, x_: o[:, 1] + (o[:, 3] ** 2 + o[:, 4] ** 2 + o[:, 5] ** 2) - f_src,
        )
        sections["residuals"] = table

    return sections


# ---------------------------------------------------------------------------------------------------------- Torch


def run_torch(only: set[str] | None) -> dict[str, Any]:
    import torch
    from omnibias.core.planner import PlanCache
    from omnibias.torch.activations.registry import get_activation, list_activations
    from omnibias.torch.experimental import Residual, compile_planned_operator, compile_residual
    from torch import nn

    torch.set_default_dtype(torch.float64)
    gen = torch.Generator().manual_seed(0)

    def randn(*shape: int) -> Any:
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    H, T, DT = 5, 3, 2

    class Chain(nn.Sequential):
        def __init__(self, dims: list[int]) -> None:
            mods: list[nn.Module] = []
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
                mods.append(nn.Linear(a, b))
                if i < len(dims) - 2:
                    mods.append(nn.Tanh())
            super().__init__(*mods)

    class Scaled(nn.Module):  # SIREN: sin(w0 * z) on the first layer
        def __init__(self) -> None:
            super().__init__()
            self.l1, self.l2, self.out = nn.Linear(D, H), nn.Linear(H, H), nn.Linear(H, 1)

        def forward(self, x: Any) -> Any:
            h = torch.sin(7.0 * self.l1(x))
            h = torch.sin(self.l2(h))
            return self.out(h)[..., 0]

    class Residual_(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp, self.out = nn.Linear(D, H), nn.Linear(H, 1)
            self.blocks = nn.ModuleList(nn.Linear(H, H) for _ in range(2))

        def forward(self, x: Any) -> Any:
            h = torch.tanh(self.inp(x))
            for blk in self.blocks:
                h = h + torch.tanh(blk(h))
            return self.out(h)[..., 0]

    class Fourier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.B = nn.Parameter(randn(D, 4))
            self.mid, self.out = nn.Linear(8, H), nn.Linear(H, 1)

        def forward(self, x: Any) -> Any:
            z = x @ self.B
            return self.out(torch.tanh(self.mid(torch.cat([torch.sin(z), torch.cos(z)], dim=-1))))[
                ..., 0
            ]

    class Gated(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.B = nn.Parameter(randn(D, 4))
            self.u, self.v = nn.Linear(8, H), nn.Linear(8, H)
            self.w1, self.w2, self.out = nn.Linear(8, H), nn.Linear(H, H), nn.Linear(H, 1)

        def forward(self, x: Any) -> Any:
            z = x @ self.B
            phi = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
            u, v = torch.tanh(self.u(phi)), torch.tanh(self.v(phi))
            f = torch.tanh(self.w1(phi))
            h = f * u + (1 - f) * v
            f = torch.tanh(self.w2(h))
            h = f * u + (1 - f) * v
            return self.out(h)[..., 0]

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(D, T * DT, bias=False)
            self.q, self.k, self.v = (nn.Linear(DT, DT, bias=False) for _ in range(3))
            self.out = nn.Linear(T * DT, 1, bias=False)

        def forward(self, x: Any) -> Any:
            z = self.inp(x).unflatten(-1, (T, DT))
            a = torch.softmax(self.q(z) @ self.k(z).transpose(-1, -2) / DT**0.5, dim=-1)
            return self.out(torch.tanh((a @ self.v(z)).flatten(-2)))[..., 0]

    class Kan(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c = nn.Parameter(randn(D, H, 3) / D**0.5)
            self.out = nn.Linear(H, 1)

        def forward(self, x: Any) -> Any:
            freqs = torch.arange(1, 4, dtype=x.dtype)
            basis = torch.sin(x[..., None] * freqs)
            return self.out(torch.tanh(torch.einsum("...ik,ijk->...j", basis, self.c)))[..., 0]

    class HardBc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = Chain([D, H, H, 1])

        def forward(self, x: Any) -> Any:
            return x[..., 0] * (1 - x[..., 0]) * self.net(x)[..., 0]

    def as_double(m: nn.Module) -> nn.Module:
        torch.manual_seed(1)
        return m.double()

    mlp = as_double(Chain([D, H, H, 1]))
    folded_layers = [
        (randn(H, D) * 7.0 / D**0.5, 0.1 * randn(H) * 7.0, "sin"),
        (randn(H, H) / H**0.5, 0.1 * randn(H), "sin"),
        (randn(1, H) / H**0.5, 0.1 * randn(1), None),
    ]
    archs: dict[str, Any] = {
        "mlp_tanh": mlp,
        "siren_scaled": as_double(Scaled()),
        "siren_folded_layers": folded_layers,
        "residual_mlp": as_double(Residual_()),
        "fourier_features": as_double(Fourier()),
        "gated_pirate": as_double(Gated()),
        "attention": as_double(Attention()),
        "kan_edges": as_double(Kan()),
        "hard_boundary_mask": as_double(HardBc()),
    }
    x = randn(4, D) * 0.8
    S_full = randn(D, D) / D**0.5

    def forward_of(model: Any) -> Any:
        if isinstance(model, list):

            def f(z: Any) -> Any:
                a = z
                for W, b, act in model:
                    a = a @ W.T + b
                    if act is not None:
                        a = get_activation(act).forward(a)
                return a[..., 0]

            return f
        return lambda z: model(z)

    def lap_ref(model: Any, xs: Any, power: int = 1, A: Any = None) -> Any:
        f = forward_of(model)

        def point(z: Any) -> Any:
            out = f(z[None])
            return out.reshape(-1)[0]

        g = point
        for _ in range(power):
            g = (
                lambda g: lambda z: (
                    torch.func.hessian(g)(z).diagonal().sum()
                    if A is None
                    else (A * torch.func.hessian(g)(z)).sum()
                )
            )(g)
        return torch.func.vmap(g)(xs)

    def classify(op: Any) -> str:
        if op.matched:
            return "structured" if op.algorithm in STRUCTURED_ALGORITHMS else "fallback"
        return "reduced" if op.algorithm.startswith("subspace") else "fallback"

    sections: dict[str, Any] = {}

    def want(name: str) -> bool:
        return only is None or name in only

    if want("architectures"):
        ops = {
            "laplacian": {"power": 1},
            "weighted_laplacian": {"power": 1, "diffusion": S_full.numpy()},
            "biharmonic": {"power": 2},
            "triharmonic": {"power": 3},
        }
        table: dict[str, dict[str, Any]] = {}
        for aname, model in archs.items():
            row: dict[str, Any] = {}
            for oname, kw in ops.items():

                def run(model=model, kw=kw):
                    xs = x[:2] if kw["power"] == 3 else x
                    op = compile_planned_operator(
                        model, xs, operator="laplacian", cache=PlanCache(), **kw
                    )
                    A = None if "diffusion" not in kw else S_full @ S_full.T
                    return checked(
                        classify(op), op.algorithm, op(xs), lap_ref(model, xs, kw["power"], A)
                    )

                row[oname] = guarded(run)
            table[aname] = row
        sections["architectures"] = table

    if want("activations"):
        xa = randn(3, 4) * 0.6
        table = {}
        for name in list_activations():
            spec = get_activation(name)
            if spec.fastpath is None:
                table[name] = {
                    "_note": cell("not_provided", note="no closed-form derivative kernel")
                }
                continue
            g = torch.Generator().manual_seed(5)
            layers = [
                (torch.randn(6, 4, generator=g) * 0.5, torch.randn(6, generator=g) * 0.3, name),
                (torch.randn(6, 6, generator=g) * 0.4, torch.randn(6, generator=g) * 0.3, name),
                (torch.randn(1, 6, generator=g) * 0.5, torch.randn(1, generator=g) * 0.3, None),
            ]
            row = {}
            for oname, kw in (
                ("laplacian", {"power": 1}),
                ("laplacian_projected", {"power": 1, "force": "projected"}),
                ("laplacian_forward", {"power": 1, "force": "forward_laplacian"}),
                ("biharmonic", {"power": 2}),
            ):

                def run(layers=layers, kw=kw):
                    want_ = lap_ref(layers, xa, kw["power"])
                    try:
                        op = compile_planned_operator(
                            layers, xa, operator="laplacian", cache=PlanCache(), **kw
                        )
                    except ValueError as err:
                        op = compile_planned_operator(
                            layers,
                            xa,
                            operator="laplacian",
                            cache=PlanCache(),
                            allow_almost_everywhere=True,
                            **kw,
                        )
                        res = checked(classify(op), op.algorithm, op(xa), want_)
                        if res["status"] != "mismatch":
                            res["status"] = "refused"
                        res["note"] = (
                            "opt-in: allow_almost_everywhere; " + str(err).replace("\n", " ")[:200]
                        )
                        return res
                    return checked(classify(op), op.algorithm, op(xa), want_)

                row[oname] = guarded(run)

            def run_gram(layers=layers):
                blk = (xa, Residual(lambda q: q.lap[:, 0] + 0.5 * q.u[:, 0], ["value", "lap"]))
                op = compile_residual(layers, [blk], quantity="gram", cache=PlanCache())

                def r_of(flat: Any) -> Any:
                    parts, k = [], 0
                    for W, b, act in layers:
                        nW, nb = W.numel(), b.numel()
                        parts.append(
                            (flat[k : k + nW].reshape(W.shape), flat[k + nW : k + nW + nb], act)
                        )
                        k += nW + nb
                    ff = forward_of(parts)

                    def one(z: Any) -> Any:
                        pt = lambda w: ff(w[None]).reshape(-1)[0]  # noqa: E731
                        return torch.func.hessian(pt)(z).diagonal().sum() + 0.5 * pt(z)

                    return torch.func.vmap(one)(xa)

                flat = torch.cat([t.reshape(-1) for W, b, _ in layers for t in (W, b)])
                J = torch.func.jacrev(r_of)(flat)
                _, K = op()
                status = "structured" if op.algorithm in STRUCTURED_ALGORITHMS else "fallback"
                return checked(status, op.algorithm, K, J @ J.T, tol=1e-7)

            row["residual_gram"] = guarded(run_gram)
            table[name] = row
        sections["activations"] = table

    if want("vector_fields"):
        Dv = 6
        torch.manual_seed(2)

        class VecRes(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inp, self.mid, self.out = nn.Linear(Dv, 4), nn.Linear(4, 4), nn.Linear(4, Dv)

            def forward(self, x: Any) -> Any:
                h = torch.tanh(self.inp(x))
                h = h + torch.tanh(self.mid(h))
                return self.out(h)

        vec_models = {"mlp_tanh": Chain([Dv, 8, 8, Dv]).double(), "residual_mlp": VecRes().double()}
        xv, rhs = randn(3, Dv), randn(3, Dv)
        table = {}
        for aname, model in vec_models.items():
            row = {}

            def jac(z, model=model):
                return torch.func.jacrev(lambda w: model(w[None])[0])(z)

            def run_div(model=model):
                op = compile_planned_operator(model, xv, operator="divergence", cache=PlanCache())
                ref = torch.func.vmap(lambda z: jac(z, model).diagonal().sum())(xv)
                return checked(classify(op), op.algorithm, op(xv), ref)

            def run_logdet(model=model):
                op = compile_planned_operator(model, xv, operator="logdet", cache=PlanCache())
                ref = torch.func.vmap(
                    lambda z: torch.linalg.slogdet(torch.eye(Dv) + jac(z, model))[1]
                )(xv)
                return checked(classify(op), op.algorithm, op(xv), ref)

            def run_solve(model=model):
                op = compile_planned_operator(model, xv, operator="solve", cache=PlanCache())
                ref = torch.func.vmap(
                    lambda z, r: torch.linalg.solve(torch.eye(Dv) + jac(z, model), r)
                )(xv, rhs)
                return checked(classify(op), op.algorithm, op(xv, rhs), ref)

            def run_vlap(model=model):
                op = compile_planned_operator(model, xv, operator="laplacian", cache=PlanCache())
                ref = torch.func.vmap(
                    lambda z: torch.func.jacrev(torch.func.jacrev(lambda w: model(w[None])[0]))(z)
                    .diagonal(dim1=-2, dim2=-1)
                    .sum(-1)
                )(xv)
                return checked(classify(op), op.algorithm, op(xv), ref)

            row["divergence"] = guarded(run_div)
            row["logdet"] = guarded(run_logdet)
            row["solve"] = guarded(run_solve)
            row["vector_laplacian"] = guarded(run_vlap)
            row["curl"] = cell(
                "not_provided",
                note="first-order: torch.func.jacrev(model) gives the antisymmetric part in D reverse passes",
            )
            table[aname] = row
        sections["vector_fields"] = table

    if want("derivatives"):
        Dd = 5
        net = Chain([Dd, 8, 8, 1]).double()
        xd = randn(4, Dd) * 0.8

        def hess(z):
            return torch.func.hessian(lambda w: net(w[None])[0, 0])(z)

        table = {}

        def run_mixed():
            i, j = 1, 3
            plus = np.zeros((Dd, 1))
            plus[i, 0] = plus[j, 0] = 0.5
            minus = np.zeros((Dd, 1))
            minus[i, 0] = 0.5
            minus[j, 0] = -0.5
            a = compile_planned_operator(
                net, xd, operator="laplacian", diffusion=plus, cache=PlanCache()
            )
            b = compile_planned_operator(
                net, xd, operator="laplacian", diffusion=minus, cache=PlanCache()
            )
            ref = torch.func.vmap(lambda z: hess(z)[i, j])(xd)
            return checked("composed", f"{a.algorithm}+{b.algorithm}", a(xd) - b(xd), ref)

        def run_time():
            e = np.zeros((Dd, 1))
            e[0, 0] = 1.0
            op = compile_planned_operator(
                net, xd, operator="laplacian", directions=e, cache=PlanCache()
            )
            out = op.evaluate(xd)
            ref = torch.func.vmap(lambda z: torch.func.grad(lambda w: net(w[None])[0, 0])(z)[0])(xd)
            return checked(classify(op), op.algorithm, out.aux["directional"][..., 0], ref)

        def run_subset():
            op = compile_planned_operator(
                net, xd, operator="laplacian", coordinates=[1, 2, 3], cache=PlanCache()
            )
            ref = torch.func.vmap(lambda z: hess(z)[1:4, 1:4].diagonal().sum())(xd)
            return checked(classify(op), op.algorithm, op(xd), ref)

        table["mixed_derivative"] = guarded(run_mixed)
        table["time_derivative"] = guarded(run_time)
        table["space_subset_laplacian"] = guarded(run_subset)
        table["curl"] = cell("not_provided", note="see vector_fields")
        sections["derivatives"] = table

    if want("residuals"):
        Dr = 3
        net = Chain([Dr, 8, 8, 1]).double()
        xr = randn(5, Dr) * 0.8
        a_coeff = 1.0 + 0.5 * torch.rand(5, generator=gen, dtype=torch.float64)
        f_src = randn(5)
        normal = randn(5, Dr)
        normal = normal / normal.norm(dim=1, keepdim=True)
        table = {}
        names = [n for n, _ in net.named_parameters()]

        def quantities(params: dict, z: Any, coords: Any) -> Any:
            def point(w):
                return torch.func.functional_call(net, params, (w[None],))[0, 0]

            u = point(z)
            gr = torch.func.grad(point)(z)
            Hz = torch.func.hessian(point)(z)
            lap = Hz.diagonal().sum() if coords is None else Hz[coords][:, coords].diagonal().sum()
            bil = (
                torch.func.hessian(lambda w: torch.func.hessian(point)(w).diagonal().sum())(z)
                .diagonal()
                .sum()
            )
            return torch.cat([torch.stack([u, lap, bil]), gr])

        def residual_cell(name, fn, demand, rows, *, coordinates=None, order=None):
            def run():
                block = (
                    (xr, Residual(fn, demand))
                    if order is None
                    else (xr, Residual(fn, demand), order)
                )
                op = compile_residual(
                    net, [block], quantity="gram", coordinates=coordinates, cache=PlanCache()
                )
                coords = None if coordinates is None else torch.as_tensor(coordinates)
                base = {n: p.detach() for n, p in net.named_parameters()}

                def r_of(params: dict) -> Any:
                    outs = torch.func.vmap(lambda z: quantities(params, z, coords))(xr)
                    return rows(outs, xr)

                Jd = torch.func.jacrev(r_of)(base)
                J = torch.cat([Jd[n].reshape(xr.shape[0], -1) for n in names], dim=1)
                _, K = op()
                status = "structured" if op.algorithm in STRUCTURED_ALGORITHMS else "fallback"
                return checked(status, op.algorithm, K, J @ J.T, tol=1e-7)

            table[name] = guarded(run)

        residual_cell(
            "variable_coefficient_poisson",
            lambda q: a_coeff * q.lap[:, 0] - f_src,
            ["value", "lap"],
            lambda o, x_: a_coeff * o[:, 1] - f_src,
        )
        residual_cell(
            "allen_cahn_nonlinear",
            lambda q: 0.01 * q.lap[:, 0] + q.u[:, 0] - q.u[:, 0] ** 3,
            ["value", "lap"],
            lambda o, x_: 0.01 * o[:, 1] + o[:, 0] - o[:, 0] ** 3,
        )
        residual_cell(
            "burgers_time_space",
            lambda q: q.grad[:, 0, 0] + q.u[:, 0] * q.grad[:, 1, 0] - 0.05 * q.lap[:, 0],
            ["value", "grad", "lap"],
            lambda o, x_: o[:, 3] + o[:, 0] * o[:, 4] - 0.05 * o[:, 1],
            coordinates=[1],
        )
        residual_cell(
            "boundary_normal_derivative",
            lambda q: (q.grad[:, :, 0] * normal).sum(1) - f_src,
            ["value", "grad"],
            lambda o, x_: (o[:, 3:6] * normal).sum(1) - f_src,
            order=1,
        )
        residual_cell(
            "variable_coefficient_biharmonic",
            lambda q: a_coeff * q.bilap[:, 0] - f_src,
            ["value", "grad_sq", "lap", "bilap"],
            lambda o, x_: a_coeff * o[:, 2] - f_src,
            order=4,
        )
        residual_cell(
            "gradient_norm_nonlinearity",
            lambda q: q.lap[:, 0] + q.grad_sq[:, 0] - f_src,
            ["value", "grad_sq", "lap"],
            lambda o, x_: o[:, 1] + (o[:, 3] ** 2 + o[:, 4] ** 2 + o[:, 5] ** 2) - f_src,
        )
        sections["residuals"] = table

    return sections


# ------------------------------------------------------------------------------------------------------------ CLI


def environment(backend: str) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "tolerance": TOL,
    }
    if backend == "jax":
        import jax

        env.update(
            jax=jax.__version__,
            device=str(jax.devices()[0].device_kind if jax.devices() else "cpu"),
        )
    else:
        import torch

        env.update(torch=torch.__version__)
    return env


def cmd_run(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    only = None if args.only is None else set(args.only.split(","))
    t0 = time.perf_counter()
    sections = run_jax(only) if args.backend == "jax" else run_torch(only)
    doc = {
        "format": 1,
        "backend": args.backend,
        "environment": environment(args.backend),
        "seconds": round(time.perf_counter() - t0, 1),
        "sections": sections,
    }
    out = Path(args.out) if args.out else OUTDIR / f"matrix_{args.backend}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    bad = [
        (sec, row, col, c)
        for sec, rows in sections.items()
        for row, cols in rows.items()
        for col, c in (cols.items() if "status" not in cols else [("", cols)])
        if c["status"] in ("mismatch", "error")
    ]
    for sec, row, col, c in bad:
        print(f"FAIL {sec}/{row}/{col}: {c}", file=sys.stderr)
    print(f"wrote {out}; {len(bad)} failing cells")
    return 1 if bad else 0


def _label(c: dict[str, Any], notes: list[str]) -> str:
    text = c["status"] if not c.get("algorithm") else f"{c['status']}: `{c['algorithm']}`"
    if c["status"] in ("refused", "not_provided", "mismatch", "error") and c.get("note"):
        # one footnote per reason, not per activation name
        note = re.sub(r"\['[^']*'\]", "[activation]", c["note"].strip())
        if note not in notes:
            notes.append(note)
        text += f" [{notes.index(note) + 1}]"
    return text


def _table(rows: dict[str, Any], title: str, first: str) -> tuple[list[str], float, list[str]]:
    """Markdown table of ``rows`` (row name -> column -> cell); the largest error and the footnotes it cites."""
    cols: list[str] = []
    for cells in rows.values():
        for name in cells:
            if name not in cols:
                cols.append(name)
    notes: list[str] = []
    worst = 0.0
    lines = [
        f"| {first} | " + " | ".join(c.replace("_", " ") for c in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for row, cells in rows.items():
        parts = []
        for col in cols:
            c = cells.get(col)
            if c is None:
                parts.append("")
                continue
            worst = max(worst, float(c.get("error") or 0.0))
            parts.append(_label(c, notes))
        lines.append(f"| {row.replace('_', ' ')} | " + " | ".join(parts) + " |")
    return lines, worst, notes


SECTION_TITLES = {
    "architectures": ("Scalar fields: architecture by operator", "architecture"),
    "vector_fields": ("Vector fields", "architecture"),
    "derivatives": ("Derivative combinations", "derivative"),
    "residuals": ("Residual Gram by demanded quantities", "residual"),
    "activations": ("Activation by operator (dense chain, depth 2, float64)", "activation"),
}


STATUSES = (
    "structured",
    "reduced",
    "fallback",
    "composed",
    "refused",
    "not_provided",
    "mismatch",
    "error",
)


def load_matrices() -> dict[str, Any]:
    """The committed ``docs/coverage/matrix_<backend>.json`` documents that exist, by backend."""
    docs = {}
    for backend in ("jax", "torch"):
        path = OUTDIR / f"matrix_{backend}.json"
        if path.exists():
            docs[backend] = json.loads(path.read_text())
    return docs


def iter_cells(doc: dict[str, Any]):
    """``(section, row, column, cell)`` for every cell of one backend's document (``column`` is ``""`` for a bare cell)."""
    for sec, rows in doc["sections"].items():
        for row, cols in rows.items():
            for col, c in cols.items() if "status" not in cols else [("", cols)]:
                yield sec, row, col, c


def build_page(docs: dict[str, Any]) -> str:
    """The text of ``docs/operator-coverage.md`` for ``docs`` (a pure function of the committed JSON)."""
    out = [
        "# Operator coverage matrix",
        "",
        "Generated by `scripts/generate_coverage_matrix.py` from "
        + ", ".join(f"`docs/coverage/matrix_{b}.json`" for b in docs)
        + ". Every cell builds a small problem, asks the planner to serve it, evaluates the planned program and compares the "
        "value with a dense autodiff oracle written in the generator (nested `hessian` traces of the model's own forward pass, "
        "`jacfwd`/`jacrev` Jacobians of the residual vector), so no derivative formula of the package is involved. "
        f"A cell whose relative error exceeds {TOL:g} is a `mismatch` and fails the generator.",
        "",
        "| status | meaning |",
        "|---|---|",
        "| structured | a chain program served it: projected, forward Laplacian, polylaplacian, Woodbury or stream Gram |",
        "| reduced | a generic exact engine served a model that is not a dense chain, along `min(rows, columns)` directions of its first-layer products (the subspace programs); exact, but not a chain program |",
        "| fallback | a generic exact engine served it with no reduction (nested autodiff or Taylor-mode jets); when the engine is the dense definition itself, the oracle agrees by construction |",
        "| composed | built from calls of structured programs |",
        "| refused | the planner refused it and said why; the reason is in the footnote |",
        "| not_provided | the package has no program for it; the footnote names the route that exists |",
        "",
        "A model is a *dense chain* when it is `dot -> bias -> activation` repeated and an affine readout; a literal scale such as the "
        "`w0` of a SIREN layer is folded into the weights. Skip connections, gates, attention, element-wise reads of the input and "
        "hard-boundary masks are not chains. The subspace programs apply when the input enters the model only through products "
        "with parameter matrices (for example `x @ W`), so that every input derivative lies in the row space of the stacked "
        "matrices; any other read of the input refuses with the operation that reads it.",
        "",
    ]
    for backend, doc in docs.items():
        env = doc["environment"]
        label = "JAX" if backend == "jax" else "PyTorch"
        out += [
            f"## {label}",
            "",
            f"Environment: {json.dumps({k: v for k, v in env.items() if k != 'tolerance'})}; generated in {doc['seconds']} s.",
            "",
        ]
        counts: dict[str, int] = {}
        for rows in doc["sections"].values():
            for cells in rows.values():
                for c in cells.values() if "status" not in cells else [cells]:
                    counts[c["status"]] = counts.get(c["status"], 0) + 1
        out += ["Cells: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) + ".", ""]
        for name, (title, first) in SECTION_TITLES.items():
            rows = doc["sections"].get(name)
            if not rows:
                continue
            rows = {k: (v if "status" not in v else {"status": v}) for k, v in rows.items()}
            lines, worst, notes = _table(rows, title, first)
            out += [f"### {title}", ""] + lines + [""]
            out += [f"Largest relative error in this table: {worst:.1e}.", ""]
            for i, note in enumerate(notes, 1):
                out.append(f"{i}. {note}")
            if notes:
                out.append("")
    return "\n".join(out).rstrip() + "\n"


def cmd_render(_: argparse.Namespace) -> int:
    docs = load_matrices()
    if not docs:
        raise SystemExit("no matrix_*.json found; run the generator first")
    PAGE.write_text(build_page(docs))
    print(f"wrote {PAGE}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("backend", choices=["jax", "torch"])
    run.add_argument("--out")
    run.add_argument("--only", help="comma separated sections")
    run.set_defaults(fn=cmd_run)
    ren = sub.add_parser("render")
    ren.set_defaults(fn=cmd_render)
    args = ap.parse_args()
    try:
        return int(args.fn(args))
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
