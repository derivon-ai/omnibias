# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded CPU comparison of compiled jets and independent AD baselines.

Each method/order runs in a separate, timeout-limited process. For example::

    uv run python benchmarks/jet_compiler.py --orders 2 4 --repeats 2 \
        --methods compiler manual nested jax

The optional ``torch-jet`` baseline is the ``jet-for-pytorch`` distribution:
https://github.com/f-dangel/torch-jet . It is never installed by this script.
Results are observations for this small workload, not performance guarantees.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import provenance, rss_mb, source_provenance, write_json  # noqa: E402

METHODS = ("compiler", "inductor", "analytic", "manual", "nested", "jax", "torch-jet")
RESULT_PREFIX = "JET_COMPILER_RESULT="
PROGRESS_PREFIX = "JET_COMPILER_PROGRESS="
SOURCE_PATHS = [
    "benchmarks/jet_compiler.py", "benchmarks/_common.py",
    "packages/omnibias-core/src/omnibias/core/jet_algebra.py",
    "packages/omnibias-core/src/omnibias/core/jet_systems.py",
    "packages/omnibias-core/src/omnibias/core/jet_series.py",
    "packages/omnibias-core/src/omnibias/core/multi_index.py",
    "packages/omnibias-core/src/omnibias/core/spec.py",
    "packages/omnibias-torch/src/omnibias/torch/experimental/compiler.py",
    "packages/omnibias-torch/src/omnibias/torch/experimental/__init__.py",
    "packages/omnibias-torch/src/omnibias/torch/jet.py",
    "packages/omnibias-torch/src/omnibias/torch/activations/registry.py",
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--orders", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    result.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--batch", type=int, default=16)
    result.add_argument("--hidden", type=int, default=8)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--timeout", type=float, default=90.0)
    result.add_argument("--nested-max-order", type=int, default=8)
    result.add_argument("--output", default="jet_compiler.json")
    result.add_argument("--worker", choices=METHODS, help=argparse.SUPPRESS)
    result.add_argument("--order", type=int, help=argparse.SUPPRESS)
    return result


def validate(args: argparse.Namespace) -> None:
    if any(n < 1 for n in args.orders) or (args.order is not None and args.order < 1):
        raise ValueError("derivative orders must be positive")
    if args.repeats < 1 or args.warmup < 0 or args.batch < 1 or args.hidden < 1 or args.threads < 1:
        raise ValueError("positive repeats, batch, hidden width and threads; nonnegative warmup required")
    if args.timeout <= 0 or args.nested_max_order < 1:
        raise ValueError("timeout and nested derivative budget must be positive")
    if Path(args.output).name != args.output:
        raise ValueError("output must be a filename within the benchmark artifact directory")


def model_data(seed: int, hidden: int, batch: int) -> tuple[list[Any], list[list[float]]]:
    """Shared stored float64 operands, independent of either backend RNG."""
    rng = random.Random(seed)
    layers = []
    for din, dout in zip((1, hidden, hidden), (hidden, hidden, 1), strict=True):
        weight = [[rng.gauss(0.0, 0.35) for _ in range(din)] for _ in range(dout)]
        bias = [rng.gauss(0.0, 0.1) for _ in range(dout)]
        layers.append((weight, bias))
    points = [[-0.4 + i / max(1, batch - 1)] for i in range(batch)]
    return layers, points


def reference(layers: list[Any], point: float, order: int) -> tuple[str, str]:
    """Independent 80-digit differentiation of the scalar mathematical MLP."""
    import mpmath as mp

    with mp.workdps(80):
        weights = [([[mp.mpf(v) for v in row] for row in w], [mp.mpf(v) for v in b])
                   for w, b in layers]

        def forward(value: Any) -> Any:
            state = [value]
            for i, (weight, bias) in enumerate(weights):
                state = [sum(w * x for w, x in zip(row, state, strict=True)) + b
                         for row, b in zip(weight, bias, strict=True)]
                if i < len(weights) - 1:
                    state = [mp.tanh(v) for v in state]
            return state[0]

        value = mp.mpf(point)
        return mp.nstr(forward(value), 80), mp.nstr(mp.diff(forward, value, order), 80)


def accuracy(actual: float, expected: str) -> dict[str, Any]:
    import mpmath as mp

    if not math.isfinite(actual):
        return {"status": "nonfinite", "actual": str(actual), "reference": expected}
    with mp.workdps(80):
        wanted = mp.mpf(expected)
        error = abs(mp.mpf(actual) - wanted)
        return {"status": "ok", "actual": actual, "reference": expected,
                "absolute_error": float(error),
                "relative_error": float(error / abs(wanted)) if wanted else None,
                "scaled_error": float(error / max(mp.mpf(1), abs(wanted)))}


def _synchronize(value: Any) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elif isinstance(value, (list, tuple)):
        for leaf in value:
            _synchronize(leaf)
    elif isinstance(value, dict):
        for leaf in value.values():
            _synchronize(leaf)


def timing(prepare: Callable[[], Any], action: Callable[[Any], Any], *, repeats: int,
           warmup: int) -> dict[str, Any]:
    """Preparation/reset is outside the timed region, including first call."""
    samples = []
    for iteration in range(1 + warmup + repeats):
        argument = prepare()
        start = time.perf_counter()
        _synchronize(action(argument))
        elapsed = (time.perf_counter() - start) * 1000
        if iteration == 0:
            first = elapsed
        elif iteration > warmup:
            samples.append(elapsed)
    return {"status": "ok", "first_call_ms": first, "median_ms": statistics.median(samples),
            "samples_ms": samples}


def _source(function: Any) -> dict[str, str] | None:
    filename = inspect.getsourcefile(function)
    if filename is None:
        return None
    path = Path(filename)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _progress(row: dict[str, Any], stage: str) -> None:
    """Preserve completed forward/oracle measurements if training times out."""
    print(PROGRESS_PREFIX + json.dumps(row | {"completed_stage": stage,
          "peak_process_rss_mib": rss_mb(),
          "rss_scope": "worker high-water RSS at completed checkpoint; not tensor allocation"},
          allow_nan=False), flush=True)


def _torch_worker(args: argparse.Namespace, row: dict[str, Any], layers: list[Any],
                  points: list[list[float]]) -> dict[str, Any]:
    import torch
    from torch import nn

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    model = nn.Sequential(nn.Linear(1, args.hidden), nn.Tanh(), nn.Linear(args.hidden, args.hidden),
                          nn.Tanh(), nn.Linear(args.hidden, 1)).double()
    dense = (model[0], model[2], model[4])
    with torch.no_grad():
        for module, (weight, bias) in zip(dense, layers, strict=True):
            module.weight.copy_(torch.tensor(weight))
            module.bias.copy_(torch.tensor(bias))
    params = tuple(model.parameters())
    originals = tuple(p.detach().clone() for p in params)
    x = torch.tensor(points)
    order = args.order
    # PyTorch routes a Python integer through int64 before scalar promotion;
    # factorials exceed that range from order21 despite fitting float64.
    fact = float(math.factorial(order))
    row["backend_version"] = torch.__version__
    row["threads"] = {"intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads()}
    row["capture_ms"] = None
    row["compilation_ms"] = None
    row["compilation_and_first_call_ms"] = None
    row["parameter_policy"] = "live tensors; restored to identical initial state outside every timed training sample"

    if args.worker in {"compiler", "analytic", "inductor"}:
        from omnibias.torch.experimental import JetSpec, compile_jet, directional_seed

        spec = JetSpec.directional(order)
        start = time.perf_counter()
        program = compile_jet(model, (x,), spec=spec,
                              backward="analytic" if args.worker == "analytic" else "native")
        row["capture_ms"] = (time.perf_counter() - start) * 1000
        row["program"] = program.explain()
        input_jet = directional_seed(x, torch.ones_like(x), spec)
        if args.worker == "inductor":
            start = time.perf_counter()
            program = torch.compile(program, backend="inductor", fullgraph=True)
            program(input_jet)
            row["compilation_and_first_call_ms"] = (time.perf_counter() - start) * 1000
            row["compilation_note"] = "Inductor forward compilation and first execution are inseparable; backward may compile separately"

        def evaluate() -> tuple[Any, Any]:
            coefficients = program(input_jet)
            return coefficients[0], coefficients[order] * fact

    elif args.worker == "manual":
        from omnibias.torch.jet import mlp_jet

        layer_specs = [(layer.weight, layer.bias, None if i == 2 else "tanh") for i, layer in enumerate(dense)]

        def evaluate() -> tuple[Any, Any]:
            coefficients = mlp_jet(x, torch.ones_like(x), layer_specs, order, riccati=True)
            return coefficients[0], coefficients[order] * fact

    elif args.worker == "nested":
        def evaluate() -> tuple[Any, Any]:
            coordinate = x.detach().clone().requires_grad_(True)
            value = model(coordinate)
            derivative = value
            for _ in range(order):
                derivative = torch.autograd.grad(derivative.sum(), coordinate, create_graph=True)[0]
            return value, derivative

    elif args.worker == "torch-jet":
        try:
            version = importlib.metadata.version("jet-for-pytorch")
            from jet import jet
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            return row | {"status": "unavailable", "reason": f"optional jet-for-pytorch is not installed: {exc}"}
        row["baseline_version"] = version
        row["baseline_source"] = _source(jet)

        def forward(coordinate: Any, *parameters: Any) -> Any:
            for i in range(0, len(parameters), 2):
                # Use the library's supported aten.t/addmm route; Tensor.T
                # traces to aten.permute, which jet-for-pytorch1.0 lacks.
                coordinate = torch.nn.functional.linear(coordinate, parameters[i], parameters[i+1])
                if i < len(parameters) - 2:
                    coordinate = torch.tanh(coordinate)
            return coordinate

        start = time.perf_counter()
        transformed = jet(forward, (x, *params))
        row["capture_ms"] = (time.perf_counter() - start) * 1000
        input_jet = (x, torch.ones_like(x), *([torch.zeros_like(x)] * (order - 1)))
        parameter_jets = tuple((p, *([torch.zeros_like(p)] * order)) for p in params)

        def evaluate() -> tuple[Any, Any]:
            tower = transformed(input_jet, *parameter_jets)
            return tower[0], tower[order]

    else:
        raise ValueError(f"unexpected Torch method {args.worker}")

    row["evaluation"] = timing(lambda: None, lambda _: evaluate(), repeats=args.repeats, warmup=args.warmup)
    value, derivative = evaluate()
    chosen = len(points) // 2
    primal_reference, derivative_reference = reference(layers, points[chosen][0], order)
    row["accuracy"] = {"point": points[chosen][0], "decimal_digits": 80,
                       "primal": accuracy(float(value[chosen, 0].detach()), primal_reference),
                       "derivative": accuracy(float(derivative[chosen, 0].detach()), derivative_reference)}
    _progress(row, "evaluation_and_accuracy")
    optimizer = torch.optim.SGD(params, lr=1e-4)

    def reset() -> None:
        with torch.no_grad():
            for parameter, original in zip(params, originals, strict=True):
                parameter.copy_(original)
        optimizer.zero_grad(set_to_none=True)

    def loss() -> Any:
        primal, derivative = evaluate()
        return primal.square().mean() + derivative.square().mean()

    def backward_prepare() -> Any:
        reset()
        return loss()

    row["backward_only"] = timing(backward_prepare, lambda objective: objective.backward(),
                                  repeats=args.repeats, warmup=args.warmup)
    row["backward_only"]["scope"] = "loss.backward only; fresh derivative forward graph and parameter reset excluded"

    def train_step(_: Any) -> Any:
        optimizer.zero_grad(set_to_none=True)
        objective = loss()
        objective.backward()
        optimizer.step()
        return objective

    row["training_step"] = timing(reset, train_step, repeats=args.repeats, warmup=args.warmup)
    row["training_step"]["scope"] = "derivative forward + residual loss + parameter backward + SGD; identical-state reset excluded"
    row["finite_parameter_gradients"] = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
    reset()
    row["status"] = "ok"
    return row


def _jax_worker(args: argparse.Namespace, row: dict[str, Any], layers: list[Any],
                points: list[list[float]]) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from jax.experimental.jet import jet

    jax.config.update("jax_enable_x64", True)
    parameters = tuple((jnp.asarray(w), jnp.asarray(b)) for w, b in layers)
    x = jnp.asarray(points)
    order = args.order
    row["backend_version"] = jax.__version__
    row["baseline_source"] = _source(jet)
    row["parameter_policy"] = "parameters are explicit dynamic JIT arguments; every training call receives identical initial arrays"
    row["threads"] = "CPU affinity bounded by the worker; XLA runtime thread settings also recorded in environment"

    def forward(params: Any, coordinate: Any) -> Any:
        for i, (weight, bias) in enumerate(params):
            coordinate = coordinate @ weight.T + bias
            if i < len(params) - 1:
                coordinate = jnp.tanh(coordinate)
        return coordinate

    def evaluate(params: Any, coordinate: Any) -> Any:
        # params is a dynamic argument to the enclosing compiled function,
        # not a captured concrete array available for XLA constant folding.
        series = (jnp.ones_like(coordinate),) + (jnp.zeros_like(coordinate),) * (order - 1)
        primal, derivatives = jet(lambda z: forward(params, z), (coordinate,), (series,))
        return primal, derivatives[-1]

    start = time.perf_counter()
    lowered = jax.jit(evaluate).lower(parameters, x)
    row["capture_ms"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    compiled = lowered.compile()
    row["compilation_ms"] = (time.perf_counter() - start) * 1000
    row["compilation_and_first_call_ms"] = None
    row["evaluation"] = timing(lambda: None, lambda _: compiled(parameters, x),
                               repeats=args.repeats, warmup=args.warmup)
    primal, derivative = compiled(parameters, x)
    chosen = len(points) // 2
    p_ref, d_ref = reference(layers, points[chosen][0], order)
    row["accuracy"] = {"point": points[chosen][0], "decimal_digits": 80,
                       "primal": accuracy(float(primal[chosen, 0]), p_ref),
                       "derivative": accuracy(float(derivative[chosen, 0]), d_ref)}
    _progress(row, "evaluation_and_accuracy")
    row["backward_only"] = {"status": "unavailable", "reason": "JAX pullback is measured in the combined compiled training step"}

    def objective(params: Any, coordinate: Any) -> Any:
        primal, derivative = evaluate(params, coordinate)
        return jnp.mean(primal**2) + jnp.mean(derivative**2)

    def train_step(params: Any, coordinate: Any) -> Any:
        loss, gradients = jax.value_and_grad(objective)(params, coordinate)
        updated = jax.tree.map(lambda p, g: p - 1e-4 * g, params, gradients)
        return loss, updated, gradients

    start = time.perf_counter()
    training = jax.jit(train_step).lower(parameters, x).compile()
    row["training_capture_and_compile_ms"] = (time.perf_counter() - start) * 1000
    row["training_step"] = timing(lambda: None, lambda _: training(parameters, x),
                                  repeats=args.repeats, warmup=args.warmup)
    row["training_step"]["scope"] = "compiled derivative forward + residual loss + parameter backward + functional SGD"
    _, _, gradients = training(parameters, x)
    row["finite_parameter_gradients"] = all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(gradients))
    row["status"] = "ok"
    return row


def _configure_process(threads: int) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
    # XLA can create compiler pools independently of OMP settings. Bounding
    # affinity before importing it makes the requested CPU budget meaningful.
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(available[:threads]))


def worker(args: argparse.Namespace) -> dict[str, Any]:
    _configure_process(args.threads)
    row = {"method": args.worker, "order": args.order, "status": "error"}
    if args.worker == "nested" and args.order > args.nested_max_order:
        return row | {"status": "refused", "reason": f"nested AD order exceeds explicit budget {args.nested_max_order}"}
    sources = source_provenance(SOURCE_PATHS)
    row["source_provenance"] = sources
    layers, points = model_data(args.seed, args.hidden, args.batch)
    try:
        run = _jax_worker if args.worker == "jax" else _torch_worker
        result = run(args, row, layers, points)
    except ImportError as exc:
        result = row | {"status": "unavailable", "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 -- isolated benchmark reports errors as data
        result = row | {"status": "error", "reason": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-6000:]}
    result["peak_process_rss_mib"] = rss_mb()
    result["source_provenance"] = sources
    final_hashes = source_provenance(SOURCE_PATHS)["source_sha256"]
    result["source_changed_during_cell"] = sources["source_sha256"] != final_hashes
    result["rss_scope"] = "worker process high-water RSS including imports/oracle; excludes child processes; not tensor allocation"
    result["environment"] = {key: os.environ.get(key) for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "JAX_ENABLE_X64", "JAX_PLATFORMS", "XLA_FLAGS")}
    if hasattr(os, "sched_getaffinity"):
        result["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    return result


def _worker_command(args: argparse.Namespace, method: str, order: int) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--worker", method, "--order", str(order),
            "--repeats", str(args.repeats), "--warmup", str(args.warmup), "--batch", str(args.batch),
            "--hidden", str(args.hidden), "--seed", str(args.seed), "--threads", str(args.threads),
            "--nested-max-order", str(args.nested_max_order)]


def run_case(args: argparse.Namespace, method: str, order: int) -> dict[str, Any]:
    start = time.perf_counter()
    process = subprocess.Popen(_worker_command(args, method, order), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=(os.name == "posix"))
    try:
        stdout, stderr = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, stderr = process.communicate()
        partial = {}
        for line in stdout.splitlines():
            if line.startswith(PROGRESS_PREFIX):
                try:
                    partial = json.loads(line[len(PROGRESS_PREFIX):])
                except json.JSONDecodeError:
                    pass
        return partial | {"method": method, "order": order, "status": "timeout", "timeout_seconds": args.timeout,
                          "reason": "isolated method/order exceeded its wall-time budget", "stderr_tail": stderr[-2000:]}
    candidates = [line[len(RESULT_PREFIX):] for line in stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    if process.returncode != 0 or not candidates:
        return {"method": method, "order": order, "status": "error", "returncode": process.returncode,
                "reason": "worker did not return a successful structured result", "stderr_tail": stderr[-4000:]}
    try:
        result = json.loads(candidates[-1])
    except json.JSONDecodeError as exc:
        return {"method": method, "order": order, "status": "error", "reason": f"invalid worker JSON: {exc}"}
    result["isolated_wall_seconds"] = time.perf_counter() - start
    if stderr.strip():
        result["stderr_tail"] = stderr[-2000:]
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    validate(args)
    if args.worker:
        if args.order is None:
            raise ValueError("worker requires --order")
        print(RESULT_PREFIX + json.dumps(worker(args), allow_nan=False), flush=True)
        return
    rows = []
    for order in args.orders:
        for method in args.methods:
            row = run_case(args, method, order)
            rows.append(row)
            print(f"{method} order={order}: {row['status']}", flush=True)
    # Set backend policy before provenance imports optional framework modules.
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    config = {"seed": args.seed, "dimensions": [1, args.hidden, args.hidden, 1],
              "batch_size": args.batch, "dtype": "float64", "device": "cpu",
              "orders": args.orders, "methods": args.methods, "repeats": args.repeats,
              "warmup": args.warmup, "threads": args.threads, "timeout_seconds": args.timeout,
              "nested_max_order": args.nested_max_order, "optimizer": "SGD", "learning_rate": 1e-4,
              "loss": "mean(primal^2) + mean(raw_order_n_derivative^2)",
              "training_state": "identical initial parameters for every timed sample; no momentum/state",
              "reference": "independent mpmath.diff of scalar MLP at one representative input; 80 digits",
              "timing_scope": "Torch eval retains parameter graph; JAX eval is compiled forward; training is measured separately",
              "compile_cache_policy": "fresh worker processes; persistent compiler disk caches are not cleared",
              "rss_scope": "per-worker process high-water RSS, including imports and oracle; not tensor allocation",
              "optional_baseline": "jet-for-pytorch; https://github.com/f-dangel/torch-jet"}
    payload = provenance(schema="omnibias/jet-compiler/v1", config=config)
    payload.update(source_provenance(SOURCE_PATHS))
    payload["rows"] = rows
    print(f"wrote {write_json(args.output, payload)}", flush=True)


if __name__ == "__main__":
    main()
