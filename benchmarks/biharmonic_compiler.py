# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded exact biharmonic comparison on a shared D -> H -> 1 tanh field.

The shallow workload admits an independent 80-digit ridge-field oracle. It
does not establish deep-network throughput or a universal compiler speedup.
Example: ``python benchmarks/biharmonic_compiler.py --repeats 2 --timeout 20``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import provenance, rss_mb, source_provenance, write_json  # noqa: E402
from jet_compiler import _configure_process, _source, timing  # noqa: E402

METHODS = ("packed", "packed-inductor", "support", "nested", "folx", "torch-jet")
RESULT_PREFIX = "BIHARMONIC_RESULT="
PROGRESS_PREFIX = "BIHARMONIC_PROGRESS="
SOURCE_PATHS = [
    "benchmarks/biharmonic_compiler.py", "benchmarks/jet_compiler.py",
    "benchmarks/_common.py", "benchmarks/_operator_reference.py",
    "packages/omnibias-torch/src/omnibias/torch/biharmonic.py",
    "packages/omnibias-torch/src/omnibias/torch/laplacian.py",
    "packages/omnibias-torch/src/omnibias/torch/jet.py",
    "packages/omnibias-torch/src/omnibias/torch/jet_mv.py",
    "packages/omnibias-torch/src/omnibias/torch/activations/classical.py",
    "packages/omnibias-core/src/omnibias/core/contraction.py",
    "packages/omnibias-core/src/omnibias/core/multi_index.py",
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dims", nargs="+", type=int, default=[2, 4])
    result.add_argument("--methods", nargs="+", choices=METHODS,
                        default=["packed", "support", "nested", "folx", "torch-jet"])
    result.add_argument("--hidden", type=int, default=4)
    result.add_argument("--batch", type=int, default=2)
    result.add_argument("--repeats", type=int, default=2)
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--timeout", type=float, default=20.0)
    result.add_argument("--output", default="biharmonic_compiler.json")
    result.add_argument("--worker", choices=METHODS, help=argparse.SUPPRESS)
    result.add_argument("--dim", type=int, help=argparse.SUPPRESS)
    return result


def validate(args: argparse.Namespace) -> None:
    if any(d < 1 for d in args.dims) or (args.dim is not None and args.dim < 1):
        raise ValueError("dimensions must be positive")
    if min(args.hidden, args.batch, args.repeats, args.threads) < 1 or args.warmup < 0 or args.timeout <= 0:
        raise ValueError("positive hidden, batch, repeats, threads, timeout and nonnegative warmup required")
    if Path(args.output).name != args.output:
        raise ValueError("output must be a filename within the artifact directory")


def model_data(seed: int, dim: int, hidden: int, batch: int) -> tuple[Any, Any]:
    rng = random.Random(seed)
    w = [[rng.gauss(0, 0.35) for _ in range(dim)] for _ in range(hidden)]
    b = [rng.gauss(0, 0.1) for _ in range(hidden)]
    c = [[rng.gauss(0, 0.35) for _ in range(hidden)]]
    offset = [rng.gauss(0, 0.1)]
    points = [[rng.uniform(-0.5, 0.5) for _ in range(dim)] for _ in range(batch)]
    return (w, b, c, offset), points


def state_counts(dim: int) -> dict[str, Any]:
    """Logical scalar counts, not peak tensor allocations or autodiff tapes."""
    return {"packed_components_per_scalar": 2 + 2 * dim + dim * (dim + 1) // 2,
            "full_total_degree_four_coefficients": math.comb(dim + 4, 4),
            "support_groups": dim + math.comb(dim, 2),
            "support_coefficient_evaluations_per_scalar": 5 * dim + 15 * math.comb(dim, 2),
            "largest_support_local_jet": 5 if dim == 1 else 15,
            "scope": "forward algebra counts only; group coefficient evaluations are not simultaneous storage"}


def reference(data: Any, points: Any) -> list[float]:
    import numpy as np
    from _operator_reference import tanh_field_laplacian_reference

    w, b, c, _ = data
    return tanh_field_laplacian_reference(np.asarray(points), np.asarray(w), np.asarray(b),
                                         np.asarray(c[0]), 2).tolist()


def agreement(actual: list[float], expected: list[float]) -> dict[str, Any]:
    if not all(math.isfinite(v) for v in actual):
        return {"status": "nonfinite", "actual": [str(v) for v in actual], "reference": expected}
    errors = [abs(a - b) for a, b in zip(actual, expected, strict=True)]
    return {"status": "ok",
            "actual": actual, "reference": expected, "max_absolute_error": max(errors),
            "max_scaled_error": max(e / max(1.0, abs(b)) for e, b in zip(errors, expected, strict=True)),
            "reference_note": "independent 80-digit mpmath ridge derivative, rounded to float64"}


def progress(row: dict[str, Any]) -> None:
    print(PROGRESS_PREFIX + json.dumps(row | {"completed_stage": "forward_and_accuracy",
          "peak_process_rss_mib": rss_mb(), "rss_scope": "checkpoint process RSS; not tensor allocation"},
          allow_nan=False), flush=True)


def torch_worker(args: argparse.Namespace, row: dict[str, Any], data: Any, points: Any) -> dict[str, Any]:
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    params = tuple(torch.nn.Parameter(torch.tensor(value)) for value in data)
    originals = tuple(p.detach().clone() for p in params)
    x = torch.tensor(points)
    row["backend_version"] = torch.__version__
    row["capture_ms"] = None
    row["compilation_ms"] = None
    row["compilation_and_first_call_ms"] = None

    def primal(coordinate: Any) -> Any:
        w, b, c, offset = params
        return torch.nn.functional.linear(torch.tanh(torch.nn.functional.linear(coordinate, w, b)), c, offset)

    if args.worker in {"packed", "packed-inductor"}:
        from omnibias.torch.biharmonic import mlp_biharmonic

        def packed(coordinate: Any, *parameters: Any) -> Any:
            w, b, c, offset = parameters
            state = mlp_biharmonic(coordinate, [(w, b, "tanh"), (c, offset, None)])
            return state.value, state.biharmonic

        operation = packed
        if args.worker == "packed-inductor":
            start = time.perf_counter()
            operation = torch.compile(packed, backend="inductor", fullgraph=True)
            operation(x, *params)
            row["compilation_and_first_call_ms"] = (time.perf_counter() - start) * 1000

        def evaluate() -> Any:
            return operation(x, *params)

    elif args.worker == "support":
        from omnibias.torch.laplacian import deep_field_polylaplacian

        def evaluate() -> Any:
            w, b, c, offset = params
            derivative = deep_field_polylaplacian(x, [(w, b, "tanh"), (c, offset, None)], 2, mode="support")
            return primal(x), derivative

    elif args.worker == "nested":
        def point(coordinate: Any) -> Any:
            return primal(coordinate).squeeze(-1)

        def laplacian(coordinate: Any) -> Any:
            return torch.func.hessian(point)(coordinate).diagonal().sum()

        hessian_laplacian = torch.func.hessian(laplacian)

        def evaluate() -> Any:
            hessians = torch.func.vmap(hessian_laplacian)(x)
            return primal(x), hessians.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)

    elif args.worker == "torch-jet":
        from jet import bilaplacian
        from omnibias.torch.biharmonic import mlp_biharmonic

        row["baseline_source"] = _source(bilaplacian)
        row["baseline_version"] = importlib.metadata.version("jet-for-pytorch")
        row["baseline_operator"] = "jet.bilaplacian(collapsed=True, randomization=None), public exact operator"
        groups = [args.dim, args.dim * (args.dim - 1), args.dim * (args.dim - 1) // 2]
        row["collapsed_direction_groups"] = [count for count in groups if count]
        row["collapsed_components_per_scalar_per_group"] = [2 + 3 * count for count in groups if count]
        start = time.perf_counter()
        transformed = bilaplacian(primal, (x[0],), collapsed=True, randomization=None)
        row["capture_ms"] = (time.perf_counter() - start) * 1000

        def evaluate() -> Any:
            return primal(x), torch.func.vmap(transformed)(x)

        # The public operator accepts one spatial tensor and captures live
        # parameters. Check that capture preserves derivative-only gradients;
        # a primal term in the training loss alone could conceal detachment.
        derivative = evaluate()[1]
        w, b, c, offset = params
        packed = mlp_biharmonic(x, [(w, b, "tanh"), (c, offset, None)]).biharmonic
        actual_grads = torch.autograd.grad(derivative.sum(), params, allow_unused=True)
        expected_grads = torch.autograd.grad(packed.sum(), params, allow_unused=True)
        gradient_errors = []
        for p, got, wanted in zip(params, actual_grads, expected_grads, strict=True):
            got = torch.zeros_like(p) if got is None else got
            wanted = torch.zeros_like(p) if wanted is None else wanted
            if not torch.allclose(got, wanted, rtol=1e-10, atol=1e-12):
                raise ValueError("torch-jet derivative-only parameter gradients disagree with packed reference")
            gradient_errors.append(float((got - wanted).abs().max()))
        row["derivative_parameter_gradient_max_errors"] = gradient_errors
        before = derivative.detach().clone()
        with torch.no_grad():
            params[0].add_(0.01)
        after = evaluate()[1].detach()
        with torch.no_grad():
            params[0].copy_(originals[0])
        row["captured_parameters_respond_to_updates"] = not torch.equal(before, after)
        if not row["captured_parameters_respond_to_updates"]:
            raise ValueError("torch-jet captured parameter update had no observable effect")

    else:
        raise ValueError(f"unsupported Torch method {args.worker}")

    row["evaluation"] = timing(lambda: None, lambda _: evaluate(), repeats=args.repeats, warmup=args.warmup)
    _, derivative = evaluate()
    row["agreement"] = agreement(derivative.detach().flatten().tolist(), reference(data, points))
    progress(row)
    optimizer = torch.optim.SGD(params, lr=1e-4)

    def reset() -> None:
        with torch.no_grad():
            for p, initial in zip(params, originals, strict=True):
                p.copy_(initial)
        optimizer.zero_grad(set_to_none=True)

    def objective() -> Any:
        value, derivative = evaluate()
        return value.square().mean() + derivative.square().mean()

    def backward_prepare() -> Any:
        reset()
        return objective()

    row["backward_only"] = timing(backward_prepare, lambda loss: loss.backward(),
                                  repeats=args.repeats, warmup=args.warmup)

    def train(_: Any) -> Any:
        loss = objective()
        loss.backward()
        optimizer.step()
        return loss

    row["training_step"] = timing(reset, train, repeats=args.repeats, warmup=args.warmup)
    row["finite_parameter_gradients"] = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
    row["parameter_policy"] = "live tensors; identical state restored outside each training/backward sample"
    row["status"] = "ok"
    return row


def folx_worker(args: argparse.Namespace, row: dict[str, Any], data: Any, points: Any) -> dict[str, Any]:
    import folx
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    params = tuple(jnp.asarray(v) for v in data)
    x = jnp.asarray(points)
    row["backend_version"] = jax.__version__
    row["baseline_source"] = _source(folx.forward_laplacian)

    def evaluate(parameters: Any, coordinates: Any) -> Any:
        w, b, c, offset = parameters

        def primal(coordinate: Any) -> Any:
            return (c @ jnp.tanh(w @ coordinate + b) + offset)[0]

        first = folx.forward_laplacian(primal)

        def laplacian(coordinate: Any) -> Any:
            return first(coordinate).laplacian

        second = folx.forward_laplacian(laplacian)
        return jax.vmap(primal)(coordinates), jax.vmap(lambda z: second(z).laplacian)(coordinates)

    start = time.perf_counter()
    lowered = jax.jit(evaluate).lower(params, x)
    row["capture_ms"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    compiled = lowered.compile()
    row["compilation_ms"] = (time.perf_counter() - start) * 1000
    row["compilation_and_first_call_ms"] = None
    row["evaluation"] = timing(lambda: None, lambda _: compiled(params, x), repeats=args.repeats, warmup=args.warmup)
    _, derivative = compiled(params, x)
    row["agreement"] = agreement(derivative.tolist(), reference(data, points))
    progress(row)

    def objective(parameters: Any, coordinates: Any) -> Any:
        value, derivative = evaluate(parameters, coordinates)
        return jnp.mean(value**2) + jnp.mean(derivative**2)

    def train(parameters: Any, coordinates: Any) -> Any:
        loss, gradients = jax.value_and_grad(objective)(parameters, coordinates)
        updated = jax.tree.map(lambda p, g: p - 1e-4 * g, parameters, gradients)
        return loss, updated, gradients

    start = time.perf_counter()
    training = jax.jit(train).lower(params, x).compile()
    row["training_capture_and_compile_ms"] = (time.perf_counter() - start) * 1000
    row["training_step"] = timing(lambda: None, lambda _: training(params, x), repeats=args.repeats, warmup=args.warmup)
    _, _, gradients = training(params, x)
    row["finite_parameter_gradients"] = all(bool(jnp.isfinite(g).all()) for g in gradients)
    row["backward_only"] = {"status": "unavailable", "reason": "backward is measured in the combined JIT training step"}
    row["parameter_policy"] = "parameters are explicit dynamic JIT arguments; every step receives identical initial arrays"
    row["status"] = "ok"
    return row


def worker(args: argparse.Namespace) -> dict[str, Any]:
    _configure_process(args.threads)
    row = {"method": args.worker, "dim": args.dim, "laplacian_power": 2, "status": "error",
           "state_counts": state_counts(args.dim)}
    sources = source_provenance(SOURCE_PATHS)
    row["source_provenance"] = sources
    data, points = model_data(args.seed, args.dim, args.hidden, args.batch)
    try:
        row = (folx_worker if args.worker == "folx" else torch_worker)(args, row, data, points)
    except ImportError as exc:
        row.update(status="unavailable", reason=str(exc))
    except Exception as exc:  # noqa: BLE001 -- isolated benchmark errors are data
        row.update(status="error", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-5000:])
    row["peak_process_rss_mib"] = rss_mb()
    row["rss_scope"] = "worker high-water process RSS including imports/oracle; excludes subprocesses; not tensor allocation"
    row["source_changed_during_cell"] = sources["source_sha256"] != source_provenance(SOURCE_PATHS)["source_sha256"]
    return row


def worker_command(args: argparse.Namespace, method: str, dim: int) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--worker", method, "--dim", str(dim),
            "--batch", str(args.batch), "--hidden", str(args.hidden), "--seed", str(args.seed),
            "--repeats", str(args.repeats), "--warmup", str(args.warmup), "--threads", str(args.threads)]


def run_case(args: argparse.Namespace, method: str, dim: int) -> dict[str, Any]:
    start = time.perf_counter()
    process = subprocess.Popen(worker_command(args, method, dim), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=(os.name == "posix"))
    expired = False
    try:
        stdout, stderr = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        expired = True
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, stderr = process.communicate()
    row: dict[str, Any] = {"method": method, "dim": dim, "laplacian_power": 2}
    for line in stdout.splitlines():
        prefix = next((p for p in (RESULT_PREFIX, PROGRESS_PREFIX) if line.startswith(p)), None)
        if prefix:
            try:
                row.update(json.loads(line[len(prefix):]))
            except json.JSONDecodeError:
                continue
    if expired:
        row.update(status="timeout", timeout_seconds=args.timeout, reason="isolated cell exceeded wall-time budget")
    elif process.returncode != 0 or "status" not in row:
        row.update(status="error", returncode=process.returncode, reason="worker did not return a structured result")
    row["isolated_wall_seconds"] = time.perf_counter() - start
    if stderr.strip():
        row["stderr_tail"] = stderr[-2000:]
    return row


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    validate(args)
    if args.worker:
        if args.dim is None:
            raise ValueError("worker requires --dim")
        print(RESULT_PREFIX + json.dumps(worker(args), allow_nan=False), flush=True)
        return
    rows = []
    for dim in args.dims:
        for method in args.methods:
            row = run_case(args, method, dim)
            rows.append(row)
            print(f"{method} D={dim}: {row['status']}", flush=True)
    config = {"seed": args.seed, "dims": args.dims, "hidden": args.hidden, "hidden_layers": 1,
              "batch": args.batch, "dtype": "float64", "device": "cpu", "laplacian_power": 2,
              "repeats": args.repeats, "warmup": args.warmup, "threads": args.threads,
              "timeout_seconds": args.timeout, "methods": args.methods,
              "loss": "mean(primal**2) + mean(biharmonic**2)", "optimizer": "SGD", "learning_rate": 1e-4,
              "training_state": "identical initial parameters for each sample; reset excluded from timing",
              "scope": "shallow tanh ridge field, not a deep-network or general compiler speed claim",
              "compile_cache_policy": "fresh worker processes; persistent compiler disk caches are not cleared",
              "timing_scope": "Torch forward retains parameter graph; folx forward is JIT compiled; training separate"}
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    payload = provenance(schema="omnibias/biharmonic-compiler/v1", config=config)
    payload.update(source_provenance(SOURCE_PATHS))
    payload["rows"] = rows
    print(f"wrote {write_json(args.output, payload)}", flush=True)


if __name__ == "__main__":
    main()
