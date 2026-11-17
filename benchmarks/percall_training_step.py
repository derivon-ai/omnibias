#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Latency of one PINN training step: planned operator inside a captured step versus the usual baselines.

One *step* is: draw ``n_int`` interior and ``n_bd`` boundary points, evaluate the Poisson residual
``Delta u - 1`` and the boundary residual ``u - sum(x**2)/(2D)`` of a tanh MLP, take the gradient into the parameters
and apply Adam. The step is timed as wall seconds per step of a block of ``steps_per_call`` steps (a CUDA-graph
replay on Torch, one ``lax.scan`` executable on JAX), synchronised at the ends, with compilation and capture
excluded. Methods differ only in how the interior Laplacian is produced:

* Torch: the planned operator (training plan: ``differentiate="params"``, ``mode="auto"``), collapsed Taylor mode
  (``jet.Laplacian``, simplified) and nested reverse-mode autodiff; each with and without CUDA-graph capture.
* JAX: the planned operator, folx ``forward_laplacian`` and ``jnp.trace(jax.hessian)``.

The pre-registered Torch target (T5) is a captured step of at most ``--target-ms`` milliseconds at the e2e subset
size (``D`` in 100 and 1000, width 64, three hidden layers, 128 interior and 64 boundary points, float64). The
result is reported whichever way it falls; ``meets_target`` is a measurement, not a gate.

usage::

    python benchmarks/percall_training_step.py --backend torch --out out.json [--dims 100,1000] [--dtype float64]
    python benchmarks/percall_training_step.py --backend jax   --out out.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from typing import Any

REPEATS = 7


class _Skip(Exception):
    """A method that is not run at this size (recorded, not an error)."""


def summarize(samples: list[float]) -> dict[str, float]:
    samples = sorted(samples)
    return {"ms_per_step": 1e3 * statistics.median(samples), "min_ms": 1e3 * samples[0], "max_ms": 1e3 * samples[-1],
            "repeats": len(samples)}


# ---------------------------------------------------------------------------------------------------------------- Torch
def run_torch(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    from omnibias.torch.experimental.training_step import (
        StepBlock,
        compile_training_step,
        planned_evaluator,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    elif args.precision == "highest":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    def layers_for(D: int, seed: int = 0) -> list[tuple[Any, Any]]:
        g = np.random.default_rng(seed)
        widths = [D] + [args.width] * args.depth + [1]
        out = []
        for a, b in zip(widths[:-1], widths[1:], strict=True):
            W = torch.tensor(g.standard_normal((b, a)) / np.sqrt(a), dtype=dtype, device=device, requires_grad=True)
            bias = torch.tensor(0.2 * g.standard_normal(b), dtype=dtype, device=device, requires_grad=True)
            out.append((W, bias))
        return out

    def u_fn(layers: Any, x: Any) -> Any:
        a = x
        for W, b in layers[:-1]:
            a = torch.tanh(a @ W.T + b)
        W, b = layers[-1]
        return (a @ W.T + b)[..., 0]

    def nested_lap(layers: Any, x: Any) -> Any:
        x = x.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(u_fn(layers, x).sum(), x, create_graph=True)
        total = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for i in range(x.shape[1]):
            (h,) = torch.autograd.grad(g[:, i].sum(), x, create_graph=True)
            total = total + h[:, i]
        return total

    def jet_lap_builder(layers: Any, D: int) -> Callable[[Any, Any], Any]:
        """``jet`` bakes parameter-dependent tensors into its module when it simplifies it, so a module built once holds
        the parameters of that moment.  Every call of the evaluator therefore builds a fresh one: inside a captured
        graph the build is recorded once and replayed on the current parameters.  An eager step would pay the build at
        every update, so ``jet_collapsed_eager`` records that build time and is not timed."""
        import jet  # noqa: F401
        from jet.laplacian import Laplacian
        from jet.simplify import simplify
        from torch.func import vmap

        mods: list[Any] = []
        for i, (W, b) in enumerate(layers):
            lin = torch.nn.Linear(W.shape[1], W.shape[0]).to(W.dtype).to(W.device)
            del lin.weight, lin.bias
            lin.weight = W
            lin.bias = b
            mods.append(lin)
            if i < len(layers) - 1:
                mods.append(torch.nn.Tanh())
        def evaluate(_p: Any, z: Any) -> Any:
            module = simplify(Laplacian(torch.nn.Sequential(*mods), dummy_x=torch.zeros(D, dtype=dtype, device=device)))
            return vmap(lambda y: module(y)[2].reshape(()))(z)

        return evaluate

    def step_for(D: int, method: str) -> Any:
        layers = layers_for(D)
        flat = [t for pair in layers for t in pair]
        spec = [(W, b, "tanh" if i < len(layers) - 1 else None) for i, (W, b) in enumerate(layers)]
        n_int, n_bd = args.n_int, args.n_bd
        scale = 1.0 / (2.0 * D)

        def s_in() -> Any:
            return 2 * torch.rand(n_int, D, dtype=dtype, device=device) - 1

        def s_bd() -> Any:
            x = 2 * torch.rand(n_bd, D, dtype=dtype, device=device) - 1
            side = 2 * (torch.rand(n_bd, 1, dtype=dtype, device=device) > 0.5).to(dtype) - 1
            return torch.cat([side, x[:, 1:]], dim=1)

        def r_in(_p: Any, _x: Any, lap: Any, _aux: Any) -> Any:
            return lap.reshape(-1) - 1.0

        def r_bd(_p: Any, x: Any, u: Any, _aux: Any) -> Any:
            return u.reshape(-1) - scale * (x * x).sum(-1)

        eval_bd: Callable[[Any, Any], Any] = lambda _p, z: u_fn(layers, z)  # noqa: E731
        graph: bool | str = "auto"
        if method.startswith("omnibias_planned"):
            # mode="auto": the protocol of the committed record (a find of the two leading candidates); the library's
            # training preset is now mode="find", top_k=None
            eval_in = planned_evaluator(spec, None, torch.zeros(n_int, D, dtype=dtype, device=device),
                                        steps=args.expected_uses, mode="auto")
        elif method.startswith("jet_collapsed"):
            eval_in = jet_lap_builder(layers, D)
        elif method.startswith("nested_autograd"):
            eval_in = lambda _p, z: nested_lap(layers, z)  # noqa: E731
        else:
            raise ValueError(method)
        if method.endswith("_eager"):
            graph = False
        blocks = [StepBlock("interior", s_in, r_in, evaluate=eval_in, order=2),
                  StepBlock("boundary", s_bd, r_bd, evaluate=eval_bd, order=0, weight=10.0)]
        return compile_training_step(blocks, flat, steps_per_call=args.steps_per_call, graph=graph, lr=1e-3,
                                     params_view=layers)

    methods = args.methods.split(",") if args.methods else [
        "omnibias_planned", "omnibias_planned_eager", "jet_collapsed", "jet_collapsed_eager", "nested_autograd",
        "nested_autograd_eager"]
    doc: dict[str, Any] = {
        "backend": "torch",
        "protocol": {"dtype": args.dtype, "precision": args.precision, "width": args.width, "depth": args.depth,
                     "n_int": args.n_int, "n_bd": args.n_bd, "steps_per_call": args.steps_per_call,
                     "blocks_per_sample": args.blocks, "repeats": REPEATS, "target_ms": args.target_ms,
                     "step": "sample + residuals + parameter gradient + fused Adam; wall time per step, synchronised"},
        "env": {"torch": torch.__version__, "python": platform.python_version(),
                "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"},
        "dims": {},
    }
    for D in args.dims:
        cell: dict[str, Any] = {}
        doc["dims"][str(D)] = cell
        for method in methods:
            rec: dict[str, Any] = {}
            try:
                if method == "jet_collapsed_eager":
                    lay = layers_for(D)
                    xs = torch.zeros(args.n_int, D, dtype=dtype, device=device)
                    t0 = time.perf_counter()
                    jet_lap_builder(lay, D)(None, xs)
                    rec["rebuild_s"] = time.perf_counter() - t0
                    rec["skipped"] = ("jet bakes parameter-dependent tensors into its module, so an eager step needs a "
                                      "fresh module after every update; the build above is what each step would add")
                    raise _Skip
                if method.startswith("nested_autograd") and D > args.dense_max_dim:
                    rec["skipped"] = f"nested autograd builds a D-step graph (D<={args.dense_max_dim} is run)"
                    raise _Skip
                t0 = time.perf_counter()
                ts = step_for(D, method)
                rec["build_s"] = time.perf_counter() - t0
                explain = ts.explain()
                rec["captured"] = bool(getattr(ts, "captured", False))
                rec["why_eager"] = explain.get("why_eager")
                samples = []
                for _ in range(REPEATS):
                    samples.append(ts.time_steps(n_blocks=args.blocks, warmup=1)["seconds_per_step"])
                rec.update(summarize(samples))
                hist = ts.run(1)
                rec["finite"] = bool(torch.isfinite(hist).all()) if torch.is_tensor(hist) else True
            except _Skip:
                pass
            except Exception as err:  # noqa: BLE001 - a failing method is a measurement outcome
                rec["error"] = f"{type(err).__name__}: {str(err)[:300]}"
            cell[method] = rec
            print(f"D={D:5d} {method:26s} " + (f"{rec['ms_per_step']:9.4f} ms/step captured={rec['captured']}"
                                                if "ms_per_step" in rec else rec.get("error", rec.get("skipped", ""))),
                  flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            with open(args.out, "w") as fh:  # a crash in a later method keeps the finished ones
                json.dump(doc, fh, indent=1)
        primary = cell.get("omnibias_planned", {})
        cell["T5"] = {
            "target_ms": args.target_ms,
            "omnibias_planned_ms": primary.get("ms_per_step"),
            "meets_target": (primary.get("ms_per_step") is not None and bool(primary.get("captured"))
                             and primary["ms_per_step"] <= args.target_ms),
        }
        doc["dims"][str(D)] = cell
        with open(args.out, "w") as fh:
            json.dump(doc, fh, indent=1)
    return doc


# ----------------------------------------------------------------------------------------------------------------- JAX
def run_jax(args: argparse.Namespace) -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", args.dtype == "float64")
    if args.precision:
        jax.config.update("jax_default_matmul_precision", "tensorfloat32" if args.precision == "tf32" else args.precision)
    import jax.numpy as jnp
    import numpy as np
    if args.stream_layout:
        import omnibias.jax.experimental.streams as _streams

        _streams.DEFAULT_LAYOUT = args.stream_layout
    from omnibias.jax.experimental.training_step import (
        StepBlock,
        adam,
        compile_training_step,
        planned_evaluator,
    )

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32

    def params_for(D: int, seed: int = 0) -> Any:
        g = np.random.default_rng(seed)
        widths = [D] + [args.width] * args.depth + [1]
        return [(jnp.asarray(g.standard_normal((a, b)) / np.sqrt(a), dtype), jnp.asarray(0.2 * g.standard_normal(b), dtype))
                for a, b in zip(widths[:-1], widths[1:], strict=True)]

    def f_point(p: Any, z: Any) -> Any:
        a = z
        for W, b in p[:-1]:
            a = jnp.tanh(a @ W + b)
        W, b = p[-1]
        return (a @ W + b)[0]

    def f_batch(p: Any, z: Any) -> Any:
        return jax.vmap(lambda y: f_point(p, y))(z)

    def hess_lap(p: Any, z: Any) -> Any:
        return jax.vmap(lambda y: jnp.trace(jax.hessian(lambda q: f_point(p, q))(y)))(z)

    def folx_lap(p: Any, z: Any) -> Any:
        import folx

        fl = folx.forward_laplacian(lambda y: f_point(p, y))
        return jax.vmap(lambda y: fl(y).laplacian)(z)

    def step_for(D: int, method: str) -> tuple[Any, Any]:
        params = params_for(D)
        n_int, n_bd = args.n_int, args.n_bd
        scale = 1.0 / (2.0 * D)

        def s_in(key: Any) -> Any:
            return jax.random.uniform(key, (n_int, D), dtype, -1.0, 1.0)

        def s_bd(key: Any) -> Any:
            k1, k2 = jax.random.split(key)
            x = jax.random.uniform(k1, (n_bd, D), dtype, -1.0, 1.0)
            side = jnp.where(jax.random.uniform(k2, (n_bd, 1), dtype) > 0.5, 1.0, -1.0).astype(dtype)
            return jnp.concatenate([side, x[:, 1:]], axis=1)

        def r_in(_p: Any, _x: Any, lap: Any, _aux: Any) -> Any:
            return lap.reshape(-1) - 1.0

        def r_bd(_p: Any, x: Any, u: Any, _aux: Any) -> Any:
            return u.reshape(-1) - scale * jnp.sum(x * x, axis=-1)

        if method == "omnibias_planned":
            # mode="auto": the protocol of the committed record; the library's training preset is mode="find", top_k=None
            ev = planned_evaluator(f_batch, params, jnp.zeros((n_int, D), dtype), steps=args.expected_uses, mode="auto")
        elif method == "folx_forward":
            ev = folx_lap
        elif method == "hessian_trace":
            ev = hess_lap
        else:
            raise ValueError(method)
        blocks = [StepBlock("interior", s_in, r_in, evaluate=ev, order=2),
                  StepBlock("boundary", s_bd, r_bd, evaluate=f_batch, order=0, weight=10.0)]
        ts = compile_training_step(blocks, adam(1e-3), steps_per_call=args.steps_per_call)
        return ts, params

    methods = args.methods.split(",") if args.methods else ["omnibias_planned", "folx_forward", "hessian_trace"]
    dev = jax.devices()[0]
    doc: dict[str, Any] = {
        "backend": "jax",
        "protocol": {"dtype": args.dtype, "precision": args.precision, "width": args.width, "depth": args.depth,
                     "n_int": args.n_int, "n_bd": args.n_bd, "steps_per_call": args.steps_per_call,
                     "blocks_per_sample": args.blocks, "repeats": REPEATS,
                     "step": "sample + residuals + parameter gradient + Adam inside one lax.scan executable"},
        "env": {"jax": jax.__version__, "python": platform.python_version(), "device": dev.device_kind,
                "platform": dev.platform},
        "dims": {},
    }
    for D in args.dims:
        cell: dict[str, Any] = {}
        for method in methods:
            rec: dict[str, Any] = {}
            try:
                if method == "hessian_trace" and D > args.dense_max_dim:
                    rec["skipped"] = f"dense Hessian at D={D} exceeds the memory budget (D<={args.dense_max_dim})"
                else:
                    t0 = time.perf_counter()
                    ts, params = step_for(D, method)
                    state = ts.init(params, 0)
                    state, _, _ = ts.run(state, 1, block_until_ready=True)  # compile
                    rec["compile_s"] = time.perf_counter() - t0
                    samples = []
                    for r in range(REPEATS):
                        state = ts.init(jax.tree.map(jnp.copy, params_for(D)), r)
                        samples.append(ts.time_steps(state, n_blocks=args.blocks, warmup=1)["seconds_per_step"])
                    rec.update(summarize(samples))
            except Exception as err:  # noqa: BLE001
                rec["error"] = f"{type(err).__name__}: {str(err)[:300]}"
            cell[method] = rec
            print(f"D={D:5d} {method:26s} " + (f"{rec['ms_per_step']:9.4f} ms/step" if "ms_per_step" in rec
                                                else rec.get("error", rec.get("skipped", ""))), flush=True)
            jax.clear_caches()
        doc["dims"][str(D)] = cell
        with open(args.out, "w") as fh:
            json.dump(doc, fh, indent=1)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("jax", "torch"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dims", default="100,1000", help="comma separated input dimensions")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3, help="hidden layers")
    ap.add_argument("--n-int", type=int, default=128)
    ap.add_argument("--n-bd", type=int, default=64)
    ap.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    ap.add_argument("--precision", default=None, help="float32 matmul policy: highest or tf32 (Torch and JAX)")
    ap.add_argument("--steps-per-call", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=20, help="calls per timing sample")
    ap.add_argument("--expected-uses", type=int, default=1000, help="uses the planner may amortise over")
    ap.add_argument("--target-ms", type=float, default=1.0)
    ap.add_argument("--dense-max-dim", type=int, default=200)
    ap.add_argument("--methods", default=None)
    ap.add_argument("--stream-layout", choices=("stack", "split"), default=None,
                    help="JAX: layout of the stream engine for every omnibias program of the run (default: the engine's default)")
    args = ap.parse_args()
    args.dims = [int(d) for d in args.dims.split(",") if d]
    run_torch(args) if args.backend == "torch" else run_jax(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
