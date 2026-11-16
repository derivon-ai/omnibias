# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Lowering probe: why one Laplacian program is faster than another at latency-bound sizes.

Per grid cell ``(D, H, L, B)`` and per method it records, for the forward pass and for value-and-gradient with respect to
the parameters:

* **JAX** -- the compiled HLO's launch count (instructions of the entry computation that start a kernel), fusions, GEMMs
  (``dot`` and cuBLAS/Triton GEMM custom calls), dense-solver calls (cuSOLVER / LAPACK custom calls), ``conditional`` and
  ``while`` instructions, host transfers and callbacks.
* **PyTorch** -- CUDA kernel launches and device-to-host synchronisations from one profiled call
  (``torch.cuda.set_sync_debug_mode`` counts the synchronising operations that the program triggers).
* steady wall-clock per call (median of ``REPEATS`` blocks of calls lasting at least ``TARGET_S``), compilation excluded.

Methods are the omnibias planner default and every program in its pool, the dense Hessian trace, forward-over-reverse,
collapsed Taylor mode (``jax.experimental.jet`` / ``jet``), folx (dense and sparsity-propagating, if installed), and the
stochastic estimators.  Nothing is dropped: a method that raises or exceeds its cap is recorded as such.

``--order 3`` times the third partial derivative ``u_xxx`` along the first input (the KdV term) and ``--order 4`` the
bi-Laplacian ``Delta^2 u``, each against its own baselines (collapsed Taylor mode, nested autodiff, nested folx or
``jet.Bilaplacian``); ``--order 2`` (default) is the Laplacian. The planner appears twice: ``omnibias_default`` is planned
for a forward evaluation and ``omnibias_default_train`` for value-and-gradient into the parameters
(``differentiate="params"``, ``mode="auto"``, ``expected_uses="auto"``).

usage::

    python benchmarks/lowering_probe.py --backend jax   --out out.json [--cells D:H:L:B,...] [--dtype float64] [--order 2|3|4]
    python benchmarks/lowering_probe.py --backend torch --out out.json [--cells ...] [--device cuda]

The default grid is ``H in {64, 256}``, ``B in {32, 256, 2048}``, ``D in {2, 16, 64, 256, 1024, 4096}`` (the H=256 rows use
a subset), ``L = 2``.  Timing runs belong on an exclusive GPU; the HLO counts are device independent in structure.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import statistics
import sys
import time
import warnings
from collections.abc import Callable
from typing import Any

#: Largest ``D`` at which the collapsed-Taylor baseline of order four is built.  The exact rational polarization plan used to be
#: dense and did not finish at ``D = 256``; the sparse plan builds there in seconds (``core.polarization.bilap.D256.scale`` of
#: ``omnibias_experiments/perf_audit_20261026``), and nothing larger was tried.
JET_POLARIZATION_MAX_DIM_ORDER4 = 256


def method_selected(name: str, spec: str) -> bool:
    """`--methods` entries are name prefixes; an entry that ends with `$` names exactly one method."""
    return any(name == m[:-1] if m.endswith("$") else name.startswith(m) for m in spec.split(",") if m)


def planner_budget(args: Any) -> float:
    """The memory budget of the planned methods: unlimited unless ``--memory-budget-gib`` says otherwise (D11)."""
    gib = getattr(args, "memory_budget_gib", 0.0)
    return float(gib) * 2**30 if gib else float("inf")


REPEATS = 9
TARGET_S = 0.03
SLOW_S = 30.0
UNROLL = 4  # calls unrolled per scan iteration in the amortized (device-time) JAX mode

#: Opcodes of an HLO entry computation that do not start a kernel.
_NO_LAUNCH = frozenset({"parameter", "get-tuple-element", "tuple", "bitcast", "constant", "after-all", "partition-id"})
_SOLVER = re.compile(r"cusolver|potrf|geqrf|orgqr|getrf|gesvd|syevd|lapack|trsm|triangular_solve|qr|cholesky", re.I)
_GEMM = re.compile(r"gemm|cublas|cublaslt|triton_gemm|__cudnn\$|matmul", re.I)
_HOST = re.compile(r"callback|host|infeed|outfeed", re.I)


def default_cells() -> list[tuple[int, int, int, int]]:
    cells = [(D, 64, 2, B) for B in (32, 256, 2048) for D in (2, 16, 64, 256, 1024, 4096)]
    cells += [(D, 256, 2, B) for B in (32, 2048) for D in (16, 256, 4096)]
    return cells


def parse_cells(text: str) -> list[tuple[int, int, int, int]]:
    out = []
    for item in text.split(","):
        if item:
            D, H, L, B = (int(v) for v in item.split(":"))
            out.append((D, H, L, B))
    return out


# --------------------------------------------------------------------------------------------------------- HLO analysis
def _opcode(rest: str) -> str | None:
    """The opcode of ``rest`` = text after ``name = ``: skips the (possibly tuple) result shape."""
    i = 0
    if rest.startswith("("):
        depth = 0
        end = len(rest) - 1
        for i, ch in enumerate(rest):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0:
                end = i
                break
        rest = rest[end + 1 :]
    m = re.search(r"\s([a-z][a-z0-9\-_]*)\(", " " + rest)
    return m.group(1) if m else None


def hlo_counts(text: str) -> dict[str, Any]:
    """Structural counts of one compiled HLO module (post-optimization text)."""
    computations: dict[str, list[tuple[str, str]]] = {}
    entry = None
    current: list[tuple[str, str]] | None = None
    for line in text.splitlines():
        s = line.strip()
        head = re.match(r"^(ENTRY\s+)?%?([\w.\-]+)\s+(\([^{]*\)\s*->.*)?\{\s*$", s)
        if head and "=" not in s.split("{")[0].split("(")[0]:
            current = computations.setdefault(head.group(2), [])
            if head.group(1):
                entry = head.group(2)
            continue
        if s == "}":
            current = None
            continue
        if current is None or " = " not in s:
            continue
        s = re.sub(r"^ROOT\s+", "", s)
        _, rest = s.split(" = ", 1)
        op = _opcode(rest)
        if op:
            current.append((op, rest))
    allops = [(op, rest) for body in computations.values() for op, rest in body]
    entry_ops = computations.get(entry or "", [])
    target = re.compile(r'custom_call_target="([^"]+)"')
    launches = [(op, rest) for op, rest in entry_ops if op not in _NO_LAUNCH]
    custom = [target.search(rest).group(1) if target.search(rest) else "" for op, rest in allops if op == "custom-call"]
    gemm_fusions = sum(1 for op, rest in allops if op == "fusion" and ("triton_gemm" in rest or "__cudnn" in rest))
    return {
        "launches": len(launches),
        "fusions": sum(1 for op, _ in launches if op == "fusion"),
        "gemms": sum(1 for op, _ in allops if op == "dot")
        + sum(1 for t in custom if _GEMM.search(t))
        + gemm_fusions,
        "solver_calls": sum(1 for t in custom if _SOLVER.search(t)),
        "conditionals": sum(1 for op, _ in allops if op == "conditional"),
        "whiles": sum(1 for op, _ in allops if op == "while"),
        "host_transfers": sum(1 for op, _ in allops if op in ("infeed", "outfeed", "send", "recv"))
        + sum(1 for t in custom if _HOST.search(t)),
        "custom_call_targets": sorted(set(t for t in custom if t)),
        "entry_opcodes": _histogram(op for op, _ in launches),
    }


def _histogram(ops: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for op in ops:
        out[op] = out.get(op, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ------------------------------------------------------------------------------------------------------------- timing
def time_blocks(fn: Callable[[], Any], block: Callable[[Any], Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    block(fn())
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    block(fn())
    second = time.perf_counter() - t0
    k = min(2000, max(1, int(TARGET_S / max(second, 1e-7))))
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = None
        for _ in range(k):
            out = fn()
        block(out)
        times.append((time.perf_counter() - t0) / k)
    times.sort()
    return {"first": first, "median": statistics.median(times), "min": times[0], "max": times[-1], "calls": k}


# ----------------------------------------------------------------------------------------------------------------- JAX
ESTIMATORS = ("hutchinson", "stde", "sdgd")  # unbiased stochastic estimators are not exact baselines
REFERENCE_PREFERENCE = ("jax_nested_hessian", "jax_hessian_trace", "jax_nested_jvp", "torch_nested_hessian",
                        "torchfunc_hessian_trace", "torch_nested_jvp", "jet_collapsed", "jet_collapsed_bilaplacian",
                        "omnibias_streams")


def attach_parity(cell: dict[str, Any], values: dict[str, Any]) -> None:
    """Record each method's forward agreement with the first available exact reference of this cell.

    Reference preference: dense nested autodiff when it fits, else collapsed Taylor mode, else the stream engine.
    A fast method that disagrees with the reference is a bug, not a win; the relative error is the evidence.
    """
    import numpy as np

    ref_name = next((n for n in REFERENCE_PREFERENCE if n in values), next(iter(values), None))
    if ref_name is None:
        return
    ref = values[ref_name]
    scale = max(float(np.max(np.abs(ref))), 1e-300)
    for name, v in values.items():
        err = float(np.max(np.abs(v - ref))) / scale if v.shape == ref.shape else float("nan")
        cell["methods"][name]["parity"] = {"reference": ref_name, "max_rel_err": err}


def run_jax(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", args.dtype == "float64")
    if args.precision:
        jax.config.update(
            "jax_default_matmul_precision", "tensorfloat32" if args.precision == "tf32" else args.precision
        )
    import jax.numpy as jnp
    from jax.experimental.jet import jet

    if getattr(args, "stream_layout", ""):
        import omnibias.jax.experimental.streams as _streams

        _streams.DEFAULT_LAYOUT = args.stream_layout  # every omnibias program of the run that goes through the stream engine
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32

    def build(D: int, H: int, L: int, B: int, seed: int = 0) -> tuple[Any, Any]:
        g = np.random.default_rng(seed)
        widths = [D] + [H] * L + [1]
        params = [
            (jnp.asarray(g.standard_normal((a, b)) / np.sqrt(a), dtype), jnp.asarray(0.2 * g.standard_normal(b), dtype))
            for a, b in zip(widths[:-1], widths[1:], strict=True)
        ]
        return params, jnp.asarray(g.standard_normal((B, D)), dtype)

    def f_point(p: Any, z: Any) -> Any:
        a = z
        for W, b in p[:-1]:
            a = jnp.tanh(a @ W + b)
        W, b = p[-1]
        return (a @ W + b)[0]

    def f_batch(p: Any, z: Any) -> Any:
        a = z
        for W, b in p[:-1]:
            a = jnp.tanh(a @ W + b)
        W, b = p[-1]
        return (a @ W + b)[:, 0]

    def high_order_methods(params: Any, x: Any, dense_ok: bool, lazy_planned: Any) -> dict[str, dict[str, Any]]:
        """Order 3 (``u_xxx`` along input 0) and order 4 (the bi-Laplacian): omnibias engines and their baselines."""
        from omnibias.core.demand import Demand, QuantitySpec, partial_requests
        from omnibias.core.polarization import polarization_plan

        D = x.shape[1]
        order = args.order
        if order == 3:
            demand, spec = Demand.of("third"), QuantitySpec(third=((3,) + (0,) * (D - 1),))
        else:
            demand, spec = Demand.of("bilap"), QuantitySpec()
        alphas, _, weights = partial_requests(demand, spec, D)
        wv = jnp.asarray(weights if order == 4 else [1.0], x.dtype)
        out: dict[str, dict[str, Any]] = {}

        def layer_list(p: Any) -> Any:
            return [(W.T, b, "tanh" if i < len(p) - 1 else None) for i, (W, b) in enumerate(p)]

        def combine(d: Any) -> Any:
            return jnp.einsum("r,br->b", wv, d[..., 0])

        def streams(p: Any, z: Any) -> Any:
            from omnibias.jax.experimental.streams import partial_derivatives

            return combine(partial_derivatives(z, layer_list(p), alphas)[1])

        def fused(p: Any, z: Any) -> Any:
            from omnibias.jax.experimental.fused_streams import fused_partial_derivatives

            return combine(fused_partial_derivatives(z, layer_list(p), alphas)[1])

        out["omnibias_streams"] = {"build": lambda: streams, "exact": True}
        out["omnibias_fused"] = {"build": lambda: fused, "exact": True}
        if order == 4:
            out["omnibias_default"] = {"build": lazy_planned(None, False, 2), "exact": True, "planned": True}
            out["omnibias_default_train"] = {"build": lazy_planned(None, True, 2), "exact": True, "planned": True}

        def build_jet_taylor() -> Callable[[Any, Any], Any]:
            # The exact rational polarization plan of order four is sparse (it used to be dense and did not finish in 20 minutes at
            # D = 256, holding more than 150 GB); it is built here, on request, not while the methods are listed.
            if order == 4 and D > JET_POLARIZATION_MAX_DIM_ORDER4:
                raise RuntimeError(
                    f"the order-four polarization plan was not tried above D = {JET_POLARIZATION_MAX_DIM_ORDER4}"
                )
            plan = polarization_plan(alphas, D)
            V = jnp.asarray(plan.direction_matrix().astype(np.float64).T, x.dtype)  # (D, K)
            # sum_r wv_r sum_{k,p} coef[r, k, p] d[k, p] with the request weights folded into one (K, n) table; the plan is
            # sparse (a dense (R, K, n) table has 8.6e9 entries at D = 256), so the fold is a scatter-add of its nonzeros
            fold = np.zeros((plan.num_directions, plan.order))
            np.add.at(fold, (plan.cols, plan.slots), np.asarray(weights if order == 4 else [1.0], dtype=np.float64)[plan.rows] * plan.values)
            fold_j = jnp.asarray(fold, x.dtype)  # (K, n)

            def jet_taylor(p: Any, z: Any) -> Any:
                def point(y: Any) -> Any:
                    def along(v: Any) -> Any:
                        series = (v,) + tuple(jnp.zeros_like(v) for _ in range(plan.order - 1))
                        _, terms = jet(lambda w: f_point(p, w), (y,), (series,))
                        return jnp.stack(terms)  # (n,)

                    d = jax.vmap(along, in_axes=1)(V)  # (K, n)
                    return jnp.einsum("kp,kp->", fold_j[:, : d.shape[1]], d)

                return jax.vmap(point)(z)

            return jet_taylor

        out["jax_jet_taylor"] = {"build": build_jet_taylor, "exact": True}

        if order == 3:

            def nested_jvp(p: Any, z: Any) -> Any:
                e = jnp.zeros(D, x.dtype).at[0].set(1.0)
                g = lambda w: f_point(p, w)  # noqa: E731
                for _ in range(3):
                    g = (lambda h: lambda w: jax.jvp(h, (w,), (e,))[1])(g)
                return jax.vmap(g)(z)

            out["jax_nested_jvp"] = {"build": lambda: nested_jvp, "exact": True}
        elif dense_ok:

            def nested_dense(p: Any, z: Any) -> Any:
                lap = lambda w: jnp.trace(jax.hessian(lambda q: f_point(p, q))(w))  # noqa: E731
                return jax.vmap(lambda y: jnp.trace(jax.hessian(lap)(y)))(z)

            out["jax_nested_hessian"] = {"build": lambda: nested_dense, "exact": True}

            def folx_nested(p: Any, z: Any) -> Any:
                import folx

                inner = folx.forward_laplacian(lambda y: f_point(p, y))
                outer = folx.forward_laplacian(lambda y: inner(y).laplacian)
                return jax.vmap(lambda y: outer(y).laplacian)(z)

            out["folx_nested"] = {"build": lambda: folx_nested, "exact": True}
        return out

    def methods(params: Any, x: Any, dense_ok: bool) -> dict[str, dict[str, Any]]:
        D = x.shape[1]
        eye = jnp.eye(D, dtype=x.dtype)
        out: dict[str, dict[str, Any]] = {}

        def planned(force: str | None, training: bool = False, power: int = 1) -> Callable[[Any, Any], Any]:
            from omnibias.core.planner import PlanCache
            from omnibias.jax.experimental import compile_operator

            extra: dict[str, Any] = {}
            if training:
                extra = {"differentiate": "params", "mode": "auto", "expected_uses": "auto", "compile_policy": "auto"}
            op = compile_operator(f_batch, (params, x), operator="laplacian", power=power, force=force,
                                  cache=PlanCache(), memory_budget=planner_budget(args), **extra)
            fn = lambda p, z: op(p, z)  # noqa: E731
            fn.algorithm = op.plan.algorithm  # type: ignore[attr-defined]
            return fn

        def lazy_planned(force: str | None, training: bool = False, power: int = 1) -> Callable[[], Callable[[Any, Any], Any]]:
            return lambda: planned(force, training, power)

        if args.order != 2:
            return high_order_methods(params, x, dense_ok, lazy_planned)
        out["omnibias_default"] = {"build": lazy_planned(None), "exact": True, "planned": True}
        out["omnibias_default_train"] = {"build": lazy_planned(None, True), "exact": True, "planned": True}
        try:
            from omnibias.core.planner import PlanCache
            from omnibias.jax.experimental import compile_operator

            op0 = compile_operator(f_batch, (params, x), operator="laplacian", cache=PlanCache(), memory_budget=float("inf"))
            for cand in [c["name"] for c in op0.plan.candidates if c["name"] not in ESTIMATORS]:
                out[f"omnibias:{cand}"] = {"build": lazy_planned(cand), "exact": True, "planned": True, "forced": cand}
        except Exception as err:  # noqa: BLE001 - recorded, not hidden
            out["omnibias:pool"] = {"error": f"{type(err).__name__}: {str(err)[:200]}"}

        def deep_field(p: Any, z: Any) -> Any:
            from omnibias.jax.laplacian import deep_field_laplacian

            layers = [(W.T, b, "tanh" if i < len(p) - 1 else None) for i, (W, b) in enumerate(p)]
            return deep_field_laplacian(z, layers)[:, 0]

        out["omnibias_deep_field"] = {"build": lambda: deep_field, "exact": True}

        def streams(form: str, layout: str = "stack") -> Callable[[], Callable[[Any, Any], Any]]:
            def build_fn() -> Callable[[Any, Any], Any]:
                from omnibias.jax.experimental.streams import second_order_streams

                def run(p: Any, z: Any) -> Any:
                    layers = [(W.T, b, "tanh" if i < len(p) - 1 else None) for i, (W, b) in enumerate(p)]
                    return second_order_streams(z, layers, form=form, layout=layout).laplacian[:, 0]

                return run

            return build_fn

        for form in ("auto", "diag", "gram"):
            out[f"omnibias_streams_{form}"] = {"build": streams(form), "exact": True}
        out["omnibias_streams_split"] = {"build": streams("diag", "split"), "exact": True}

        def fused_laplacian(block: int | None = None, warps: int | None = None) -> Callable[[], Callable[[Any, Any], Any]]:
            def build_fn() -> Callable[[Any, Any], Any]:
                from omnibias.jax.experimental.fused_streams import fused_second_order

                def run(p: Any, z: Any) -> Any:
                    layers = [(W.T, b, "tanh" if i < len(p) - 1 else None) for i, (W, b) in enumerate(p)]
                    return fused_second_order(z, layers, block=block, warps=warps).laplacian[:, 0]

                return run

            return build_fn

        out["omnibias_fused"] = {"build": fused_laplacian(), "exact": True}
        for blk, wp in ((16, 2), (32, 4), (64, 8)):
            out[f"omnibias_fused_b{blk}w{wp}"] = {"build": fused_laplacian(blk, wp), "exact": True}

        if dense_ok:
            out["jax_hessian_trace"] = {
                "build": lambda: (lambda p, z: jax.vmap(lambda y: jnp.trace(jax.hessian(lambda w: f_point(p, w))(y)))(z)),
                "exact": True,
            }

            def fwd_rev(p: Any, z: Any) -> Any:
                grad = jax.grad(lambda w: f_point(p, w))

                def lap(y: Any) -> Any:
                    return jax.vmap(lambda e: jax.jvp(grad, (y,), (e,))[1] @ e)(eye).sum()

                return jax.vmap(lap)(z)

            out["jax_forward_over_reverse"] = {"build": lambda: fwd_rev, "exact": True}

        def jet_taylor(p: Any, z: Any) -> Any:
            def lap(y: Any) -> Any:
                def second(e: Any) -> Any:
                    _, (_, d2) = jet(lambda w: f_point(p, w), (y,), ((e, jnp.zeros_like(e)),))
                    return d2

                return jax.vmap(second)(eye).sum()

            return jax.vmap(lap)(z)

        out["jax_jet_taylor"] = {"build": lambda: jet_taylor, "exact": True}

        def folx_builder(threshold: float) -> Callable[[], Callable[[Any, Any], Any]]:
            def build_fn() -> Callable[[Any, Any], Any]:
                import folx

                def run(p: Any, z: Any) -> Any:
                    fl = folx.forward_laplacian(lambda y: f_point(p, y), sparsity_threshold=threshold)
                    return jax.vmap(lambda y: fl(y).laplacian)(z)

                return run

            return build_fn

        out["folx_forward"] = {"build": folx_builder(0), "exact": True}
        out["folx_sparse"] = {"build": folx_builder(0.6), "exact": True}
        return out

    def amortized(fn: Callable[[Any, Any], Any], params: Any, x: Any, grad: bool, n: int) -> dict[str, Any]:
        """``n`` independent calls in one executable: dispatch is paid once, kernels run back to back.

        ``n`` calls are traced as a ``lax.scan`` of ``n // UNROLL`` iterations that each unroll ``UNROLL`` calls (the
        compile cost grows with ``UNROLL`` only; the loop overhead is shared by ``UNROLL`` calls).  Each call sees its
        own input and its own ``optimization_barrier`` copy of the parameters, so nothing is hoisted or shared between
        calls; every output (and, for ``grad``, every parameter gradient) is reduced into the result.
        """
        unroll = min(UNROLL, n)
        steps = max(n // unroll, 1)
        n = steps * unroll
        xs = jnp.stack([x * (1.0 + 1e-3 * i) for i in range(n)]).reshape(steps, unroll, *x.shape)

        def many(p: Any, xs: Any) -> Any:
            def body(total: Any, xb: Any) -> tuple[Any, None]:
                for j in range(unroll):
                    pi = jax.lax.optimization_barrier(p)
                    if grad:
                        v, g = jax.value_and_grad(lambda q, z: jnp.sum(fn(q, z)))(pi, xb[j])
                        total = total + v + sum(jnp.sum(t) for t in jax.tree_util.tree_leaves(g))
                    else:
                        total = total + jnp.sum(fn(pi, xb[j]))
                return total, None

            return jax.lax.scan(body, jnp.zeros((), x.dtype), xs)[0]

        jitted = jax.jit(many)
        t0 = time.perf_counter()
        jax.block_until_ready(jitted(params, xs))
        compile_s = time.perf_counter() - t0
        t = time_blocks(lambda: jitted(params, xs), jax.block_until_ready)
        return {"per_call": t["median"] / n, "n": n, "unroll": unroll, "compile_s": compile_s, "min": t["min"] / n,
                "max": t["max"] / n}

    def analyse(fn: Callable[[Any, Any], Any], params: Any, x: Any) -> dict[str, Any]:
        rec: dict[str, Any] = {}
        fwd = jax.jit(fn)
        vag = jax.jit(jax.value_and_grad(lambda p, z: jnp.sum(fn(p, z))))
        for tag, jitted in (("forward", fwd), ("value_and_grad", vag)):
            try:
                t0 = time.perf_counter()
                compiled = jitted.lower(params, x).compile()
                compile_s = time.perf_counter() - t0
                sub: dict[str, Any] = {"compile_s": compile_s, **hlo_counts(compiled.as_text())}
                t0 = time.perf_counter()
                first = jax.block_until_ready(jitted(params, x))
                if tag == "forward":
                    rec["_value"] = np.asarray(first, dtype=np.float64).reshape(-1)
                if time.perf_counter() - t0 > SLOW_S:
                    sub["slow"] = "first call over cap"
                else:
                    sub["timing"] = time_blocks(lambda j=jitted: j(params, x), jax.block_until_ready)
                    if args.amortize:
                        try:
                            sub["amortized"] = amortized(fn, params, x, tag == "value_and_grad", args.amortize)
                        except Exception as err:  # noqa: BLE001
                            sub["amortized"] = {"error": f"{type(err).__name__}: {str(err)[:200]}"}
            except Exception as err:  # noqa: BLE001 - a failing method is a measurement outcome
                sub = {"error": f"{type(err).__name__}: {str(err)[:300]}"}
            rec[tag] = sub
        return rec

    dev = jax.devices()[0]
    doc: dict[str, Any] = {
        "backend": "jax",
        "protocol": {"repeats": REPEATS, "target_s": TARGET_S, "dtype": args.dtype, "matmul_precision": args.precision,
                     "planned_memory_budget_gib": args.memory_budget_gib or "unlimited", "runtime_weights": True, "compilation_excluded": True,
                     "launches": "entry-computation instructions of the compiled module that start a kernel"},
        "env": {"jax": jax.__version__, "python": platform.python_version(), "device": dev.device_kind,
                "platform": dev.platform},
        "cells": [],
    }
    for D, H, L, B in args.cells:
        params, x = build(D, H, L, B)
        dense_ok = D * D * B * 8 < 3e9 and D <= 1024
        cell: dict[str, Any] = {"D": D, "H": H, "L": L, "B": B, "methods": {}}
        values: dict[str, Any] = {}
        for name, entry in methods(params, x, dense_ok).items():
            if args.methods and not method_selected(name, args.methods):
                continue
            if "error" in entry:
                cell["methods"][name] = entry
                continue
            try:
                fn = entry["build"]()
                rec = analyse(fn, params, x)
                if entry.get("planned"):
                    rec["algorithm"] = getattr(fn, "algorithm", None)
            except Exception as err:  # noqa: BLE001
                rec = {"error": f"{type(err).__name__}: {str(err)[:300]}"}
            if "_value" in rec:
                values[name] = rec.pop("_value")
            cell["methods"][name] = rec
            t = rec.get("forward", {}).get("timing", {}).get("median")
            tg = rec.get("value_and_grad", {}).get("timing", {}).get("median")
            af = rec.get("forward", {}).get("amortized", {}).get("per_call")
            ag = rec.get("value_and_grad", {}).get("amortized", {}).get("per_call")
            print(f"D={D:5d} H={H} B={B:5d} {name:34s} fwd {'' if t is None else f'{t * 1e3:9.3f}'} ms  "
                  f"vag {'' if tg is None else f'{tg * 1e3:9.3f}'} ms  amort {'' if af is None else f'{af * 1e3:8.3f}'}/"
                  f"{'' if ag is None else f'{ag * 1e3:8.3f}'} ms  launches "
                  f"{rec.get('forward', {}).get('launches')}/{rec.get('value_and_grad', {}).get('launches')} "
                  f"solver {rec.get('forward', {}).get('solver_calls')} cond {rec.get('forward', {}).get('conditionals')}",
                  flush=True)
            jax.clear_caches()
        attach_parity(cell, values)
        doc["cells"].append(cell)
        with open(args.out, "w") as fh:
            json.dump(doc, fh, indent=1)
    return doc


# ---------------------------------------------------------------------------------------------------------------- Torch
def run_torch(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch.func import hessian, jvp, vmap

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    device = args.device
    if args.precision == "highest":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    elif args.precision in ("tf32", "tensorfloat32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    def build(D: int, H: int, L: int, B: int, seed: int = 0) -> tuple[Any, Any, Any]:
        g = np.random.default_rng(seed)
        widths = [D] + [H] * L + [1]
        layers = [
            (
                torch.tensor(g.standard_normal((b, a)) / np.sqrt(a), dtype=dtype, device=device, requires_grad=True),
                torch.tensor(0.2 * g.standard_normal(b), dtype=dtype, device=device, requires_grad=True),
            )
            for a, b in zip(widths[:-1], widths[1:], strict=True)
        ]
        x = torch.tensor(g.standard_normal((B, D)), dtype=dtype, device=device)

        def f_point(z: Any) -> Any:
            a = z
            for W, b in layers[:-1]:
                a = torch.tanh(W @ a + b)
            W, b = layers[-1]
            return (W @ a + b)[0]

        spec = [(W, b, "tanh" if k < L else None) for k, (W, b) in enumerate(layers)]
        return spec, f_point, x

    def high_order_methods(spec: Any, f_point: Any, x: Any, dense_ok: bool, planned: Any) -> dict[str, dict[str, Any]]:
        """Order 3 (``u_xxx`` along input 0) and order 4 (the bi-Laplacian): omnibias engines and their baselines."""
        from omnibias.core.demand import Demand, QuantitySpec, partial_requests

        D = x.shape[1]
        order = args.order
        if order == 3:
            demand, qspec = Demand.of("third"), QuantitySpec(third=((3,) + (0,) * (D - 1),))
        else:
            demand, qspec = Demand.of("bilap"), QuantitySpec()
        alphas, _, weights = partial_requests(demand, qspec, D)
        wv = torch.tensor(weights if order == 4 else [1.0], dtype=x.dtype, device=x.device)
        out: dict[str, dict[str, Any]] = {}

        def combine(d: Any) -> Any:
            return torch.einsum("r,br->b", wv, d[..., 0])

        def streams() -> Callable[[], Any]:
            from omnibias.torch.experimental.streams import partial_derivatives

            return lambda: combine(partial_derivatives(x, spec, alphas)[1])

        def fused() -> Callable[[], Any]:
            from omnibias.torch.experimental.fused_streams import fused_partial_derivatives

            return lambda: combine(fused_partial_derivatives(x, spec, alphas)[1])

        out["omnibias_streams"] = {"build": streams}
        out["omnibias_fused"] = {"build": fused}
        if order == 4:
            out["omnibias_default"] = {"build": planned(None, False, 2)}
            out["omnibias_default_train"] = {"build": planned(None, True, 2)}

        if order == 3:

            def nested_jvp() -> Callable[[], Any]:
                e = torch.zeros(D, dtype=x.dtype, device=x.device)
                e[0] = 1.0

                def run() -> Any:
                    g = f_point
                    for _ in range(3):
                        g = (lambda h: lambda w: jvp(h, (w,), (e,))[1])(g)
                    return vmap(g)(x)

                return run

            out["torch_nested_jvp"] = {"build": nested_jvp}
        else:
            if dense_ok:

                def nested_dense() -> Callable[[], Any]:
                    lap = lambda w: hessian(f_point)(w).diagonal().sum()  # noqa: E731
                    return lambda: vmap(lambda y: hessian(lap)(y).diagonal().sum())(x)

                out["torch_nested_hessian"] = {"build": nested_dense}

            def jet_bilaplacian() -> Callable[[], Any]:
                import jet  # noqa: F401
                from jet.bilaplacian import Bilaplacian
                from jet.simplify import simplify

                mods = []
                for W, b, act in spec:
                    lin = torch.nn.Linear(W.shape[1], W.shape[0]).to(W.dtype).to(W.device)
                    del lin.weight, lin.bias
                    lin.weight = W
                    lin.bias = b
                    mods.append(lin)
                    if act == "tanh":
                        mods.append(torch.nn.Tanh())
                module = simplify(Bilaplacian(torch.nn.Sequential(*mods), dummy_x=torch.zeros(D, dtype=x.dtype, device=x.device)))

                def run() -> Any:
                    def one(z: Any) -> Any:
                        r = module(z)
                        r = r[-1] if isinstance(r, tuple) else r
                        return r.reshape(())

                    return vmap(one)(x)

                return run

            out["jet_collapsed_bilaplacian"] = {"build": jet_bilaplacian}
        return out

    def methods(spec: Any, f_point: Any, x: Any, dense_ok: bool) -> dict[str, dict[str, Any]]:
        D = x.shape[1]
        out: dict[str, dict[str, Any]] = {}

        def planned(force: str | None, training: bool = False, power: int = 1) -> Callable[[], Any]:
            def build_fn() -> Callable[[], Any]:
                from omnibias.core.planner import PlanCache
                from omnibias.torch.experimental import compile_planned_operator

                extra: dict[str, Any] = {}
                if training:
                    extra = {"differentiate": "params", "mode": "auto", "expected_uses": "auto"}
                op = compile_planned_operator(spec, x, power=power, force=force, cache=PlanCache(),
                                              memory_budget=planner_budget(args), **extra)
                return lambda: op.evaluate(x).value[:, 0]

            return build_fn

        if args.order != 2:
            return high_order_methods(spec, f_point, x, dense_ok, planned)
        out["omnibias_default"] = {"build": planned(None)}
        out["omnibias_default_train"] = {"build": planned(None, True)}
        try:
            from omnibias.core.planner import PlanCache
            from omnibias.torch.experimental import compile_planned_operator

            op0 = compile_planned_operator(spec, x, cache=PlanCache(), memory_budget=float("inf"))
            for cand in [c["name"] for c in op0.explain()["candidates"] if c["name"] not in ESTIMATORS]:
                out[f"omnibias:{cand}"] = {"build": planned(cand)}
        except Exception as err:  # noqa: BLE001
            out["omnibias:pool"] = {"error": f"{type(err).__name__}: {str(err)[:200]}"}
        if dense_ok:
            out["torchfunc_hessian_trace"] = {
                "build": lambda: (lambda: vmap(lambda z: hessian(f_point)(z).diagonal().sum())(x))
            }

        def jet_collapsed() -> Callable[[], Any]:
            import jet  # noqa: F401
            from jet.laplacian import Laplacian
            from jet.simplify import simplify

            mods = []
            for W, b, act in spec:
                lin = torch.nn.Linear(W.shape[1], W.shape[0]).to(W.dtype).to(W.device)
                del lin.weight, lin.bias  # plain tensors keep the benchmark's own autograd leaves in the graph
                lin.weight = W
                lin.bias = b
                mods.append(lin)
                if act == "tanh":
                    mods.append(torch.nn.Tanh())
            module = simplify(Laplacian(torch.nn.Sequential(*mods), dummy_x=torch.zeros(D, dtype=x.dtype, device=x.device)))
            return lambda: vmap(lambda z: module(z)[2].reshape(()))(x)

        out["jet_collapsed"] = {"build": jet_collapsed}

        def fwd_fwd() -> Callable[[], Any]:
            eye = torch.eye(D, dtype=x.dtype, device=x.device)

            def run() -> Any:
                def lap(z: Any) -> Any:
                    def second(e: Any) -> Any:
                        return jvp(lambda w: jvp(f_point, (w,), (e,))[1], (z,), (e,))[1]

                    return vmap(second)(eye).sum()

                return vmap(lap)(x)

            return run

        if dense_ok:
            out["nested_forward_mode"] = {"build": fwd_fwd}
        return out

    def count_kernels_and_syncs(call: Callable[[], Any]) -> dict[str, Any]:
        from torch.profiler import ProfilerActivity, profile

        for _ in range(2):
            call()
        sync()
        info: dict[str, Any] = {}
        if device == "cuda":
            torch.cuda.set_sync_debug_mode("warn")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                call()
                sync()
            torch.cuda.set_sync_debug_mode("default")
            info["host_syncs"] = sum("synchroniz" in str(w.message) for w in caught)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                call()
                sync()
            kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            info["launches"] = len(kernels)
            info["gemms"] = sum(bool(re.search(r"gemm|cutlass|cublas|sgemm|dgemm|ampere", e.name, re.I)) for e in kernels)
            info["solver_calls"] = sum(bool(re.search(r"potrf|geqrf|cusolver|trsm|getrf", e.name, re.I)) for e in kernels)
            info["kernel_names_top"] = _histogram(e.name[:60] for e in kernels)
        return info

    side: list[Any] = []

    def warmup_stream() -> Any:
        """One side stream for every capture: the BLAS workspace of a stream stays allocated for the life of the process, so a
        stream per capture kept several hundred MiB per capture (and gigabytes in the large cells) out of reach of
        ``empty_cache``."""
        if not side:
            side.append(torch.cuda.Stream())
        return side[0]

    def graph_replay(call: Callable[[], Any], need_bytes: int = 0) -> dict[str, Any]:
        """Device time of one call: CUDA-graph replay timed with events (host dispatch excluded); records why not capturable.

        A capture that fails for lack of memory leaves its private pool allocated, so every later capture of the process
        fails too.  The capture is therefore skipped, with the reason, when the device has less free memory than the
        call's own peak (``need_bytes``, measured on the eager calls) needs, and a failed graph is reset to free its pool.
        """
        graph = None
        try:
            torch.cuda.empty_cache()
            free = torch.cuda.mem_get_info()[0]
            if need_bytes and free < 1.2 * need_bytes + (256 << 20):
                return {"skipped": f"capture needs about {need_bytes / 2**30:.1f} GiB of pool and {free / 2**30:.1f} GiB are free"}
            stream = warmup_stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    call()
            torch.cuda.current_stream().wait_stream(stream)
            sync()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            for _ in range(10):
                graph.replay()
            sync()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(REPEATS):
                n = 200
                start.record()
                for _ in range(n):
                    graph.replay()
                end.record()
                sync()
                samples.append(start.elapsed_time(end) * 1e-3 / n)
            samples.sort()
            return {"per_call": statistics.median(samples), "min": samples[0], "max": samples[-1]}
        except Exception as err:  # noqa: BLE001 - not capturable is itself a finding
            sync()
            return {"error": f"{type(err).__name__}: {str(err)[:200]}"}
        finally:
            if graph is not None:
                graph.reset()
            gc.collect()
            torch.cuda.empty_cache()

    doc: dict[str, Any] = {
        "backend": "torch",
        "protocol": {"repeats": REPEATS, "target_s": TARGET_S, "dtype": args.dtype, "precision": args.precision,
                     "planned_memory_budget_gib": args.memory_budget_gib or "unlimited",
                     "launches": "CUDA kernels in one torch.profiler window", "host_syncs": "torch.cuda sync debug mode"},
        "env": {"torch": torch.__version__, "python": platform.python_version(),
                "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"},
        "cells": [],
    }
    for D, H, L, B in args.cells:
        spec, f_point, x = build(D, H, L, B)
        dense_ok = D * D * B * 8 < 3e9 and D <= 1024
        cell: dict[str, Any] = {"D": D, "H": H, "L": L, "B": B, "methods": {}}
        values = {}
        params = [t for W, b, _ in spec for t in (W, b)]
        for name, entry in methods(spec, f_point, x, dense_ok).items():
            if args.methods and not method_selected(name, args.methods):
                continue
            if "error" in entry:
                cell["methods"][name] = entry
                continue
            rec: dict[str, Any] = {}
            try:
                fn = entry["build"]()

                def fwd(fn: Callable[[], Any] = fn) -> Any:
                    with torch.no_grad():
                        return fn()

                # ``jet`` caches parameter-dependent tensors when its module is built, so a second backward through
                # them needs the graph kept.  Keeping it charges the baseline nothing for rebuilding the module each
                # step, which is the conservative choice for a comparison against it.
                retain = name.startswith("jet_")

                def vag(fn: Callable[[], Any] = fn, params: Any = params, retain: bool = retain) -> Any:
                    for p in params:
                        p.grad = None
                    fn().sum().backward(retain_graph=retain)
                    return params[0].grad

                for tag, call in (("forward", fwd), ("value_and_grad", vag)):
                    sub: dict[str, Any] = {}
                    try:
                        if device == "cuda":
                            torch.cuda.reset_peak_memory_stats()
                        base = torch.cuda.memory_allocated() if device == "cuda" else 0
                        t0 = time.perf_counter()
                        first = call()
                        sync()
                        if tag == "forward" and torch.is_tensor(first):
                            values[name] = first.detach().double().cpu().reshape(-1).numpy()
                        if time.perf_counter() - t0 > SLOW_S:
                            sub["slow"] = "first call over cap"
                        else:
                            sub["timing"] = time_blocks(call, lambda _o: sync())
                            sub.update(count_kernels_and_syncs(call))
                            if args.amortize and device == "cuda":
                                sub["graph"] = graph_replay(call, max(torch.cuda.max_memory_allocated() - base, 0))
                    except Exception as err:  # noqa: BLE001
                        sub = {"error": f"{type(err).__name__}: {str(err)[:300]}"}
                    rec[tag] = sub
            except Exception as err:  # noqa: BLE001
                rec = {"error": f"{type(err).__name__}: {str(err)[:300]}"}
            cell["methods"][name] = rec
            # Release this method's tensors, graphs and captured pools before the next one: the caching allocator would
            # otherwise carry them (and their fragmentation) into the largest cells, where they decide an out-of-memory.
            fn = fwd = vag = call = first = None
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                rec["held_after_gib"] = {"allocated": round(torch.cuda.memory_allocated() / 2**30, 3),
                                         "reserved": round(torch.cuda.memory_reserved() / 2**30, 3),
                                         "free": round(torch.cuda.mem_get_info()[0] / 2**30, 2)}
            t = rec.get("forward", {}).get("timing", {}).get("median")
            tg = rec.get("value_and_grad", {}).get("timing", {}).get("median")
            print(f"D={D:5d} H={H} B={B:5d} {name:34s} fwd {'' if t is None else f'{t * 1e3:9.3f}'} ms  "
                  f"vag {'' if tg is None else f'{tg * 1e3:9.3f}'} ms  syncs "
                  f"{rec.get('forward', {}).get('host_syncs')} launches {rec.get('forward', {}).get('launches')}",
                  flush=True)
        attach_parity(cell, values)
        doc["cells"].append(cell)
        with open(args.out, "w") as fh:
            json.dump(doc, fh, indent=1)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("jax", "torch"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", default="", help="D:H:L:B,... (default: the grid in the module docstring)")
    ap.add_argument("--order", type=int, choices=(2, 3, 4), default=2,
                    help="2: Laplacian; 3: u_xxx along input 0; 4: bi-Laplacian")
    ap.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    ap.add_argument("--precision", default="", help="matmul precision policy applied to every method alike")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--methods", default="", help="comma-separated method-name prefixes")
    ap.add_argument("--memory-budget-gib", type=float, default=0.0,
                    help="memory budget of the planned methods in GiB (default 0: unlimited, as registered); the planner then does not time "
                    "a candidate it estimates above it")
    ap.add_argument("--stream-layout", choices=("", "stack", "split"), default="",
                    help="JAX: layout of the stream engine for every omnibias program of the run (default: the engine's default)")
    ap.add_argument("--amortize", type=int, default=0,
                    help="also time N independent calls in one executable (JAX) / a CUDA-graph replay (torch): device time")
    args = ap.parse_args()
    args.cells = parse_cells(args.cells) if args.cells else default_cells()
    run = run_jax if args.backend == "jax" else run_torch
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
