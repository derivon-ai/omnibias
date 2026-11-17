# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Isolated, accuracy-gated comparisons of structural derivative compilation.

All derivative programs receive runtime coordinates and trainable parameters.
Coordinate seeding is INSIDE the reference program before backend compilation.
The default protocol uses three fresh processes and twenty timed samples. Failed
or budget-terminated cells never receive a speedup. Optional baselines are not
installed by this script. Outputs and exact measured sources go to artifacts/.
"""

from __future__ import annotations

import argparse
import gzip
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
import tempfile
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT, provenance, rss_mb, source_provenance, write_json  # noqa: E402
from _operator_models import high_precision, model_data, torch_model  # noqa: E402
from jet_compiler import _configure_process, accuracy, timing  # noqa: E402

FAMILIES = ("ridge", "separable", "bottleneck", "coupled")
METHODS = ("compiler", "reference", "manual", "nested", "jax-jet", "jax-nested", "folx", "torch-jet", "keras-jax", "jax-manual")
PREFIX = "OPERATOR_RESULT="
PROGRESS = "OPERATOR_PROGRESS="
SCHEMA = "omnibias/operator-compiler/v2"
REFERENCE_SCHEMA = "omnibias/operator-reference/v1"
ENGINE_STARTED = time.perf_counter()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    p.add_argument("--orders", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=["compiler", "reference", "manual"])
    p.add_argument("--execution", choices=["eager", "compiled"], default="compiled")
    p.add_argument("--activation-method", choices=["recurrence", "polynomial"], default="recurrence")
    p.add_argument("--kernel-lowering", choices=["auto", "inline", "regional", "packed"], default="auto")
    p.add_argument("--precision", choices=["source", "strict"], default="strict")
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--latent", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--rss-limit-mib", type=float, default=3072)
    p.add_argument("--nested-max-order", type=int, default=8)
    p.add_argument("--cache", choices=["persistent", "cold", "disabled"], default="persistent",
                   help="persistent: shared caches; cold: empty isolated caches with within-run reuse; disabled: diagnostic cache disabling")
    p.add_argument("--forward-only", action="store_true")
    p.add_argument("--output", default="operator_compiler.json")
    p.add_argument("--worker", choices=METHODS, help=argparse.SUPPRESS)
    p.add_argument("--family", choices=FAMILIES, help=argparse.SUPPRESS)
    p.add_argument("--order", type=int, help=argparse.SUPPRESS)
    p.add_argument("--reference-file", type=Path, help=argparse.SUPPRESS)
    return p


def validate(args: argparse.Namespace) -> None:
    positive = (args.replicates, args.repeats, args.batch, args.dim, args.hidden,
                args.latent, args.threads, args.timeout, args.rss_limit_mib, args.nested_max_order)
    if any(v < 1 for v in positive) or args.warmup < 0:
        raise ValueError("positive dimensions, samples, threads and resource budgets required")
    if any(n < 1 or n > 32 for n in args.orders) or (args.order is not None and not 1 <= args.order <= 32):
        raise ValueError("orders must be between 1 and 32")
    if Path(args.output).name != args.output:
        raise ValueError("output must be a filename within the artifact directory")


def hardware_details() -> dict[str, Any]:
    """Record the host model as well as each worker's explicit CPU affinity."""
    result: dict[str, Any] = {"logical_cpus": os.cpu_count(), "platform": sys.platform}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                result["cpu_model"] = line.partition(":")[2].strip()
                break
    if hasattr(os, "sched_getaffinity"):
        result["parent_cpu_affinity"] = sorted(os.sched_getaffinity(0))
    if hasattr(os, "sysconf"):
        try:
            result["installed_memory_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            pass
    return result


def source_paths() -> list[str]:
    roots = [REPO_ROOT / "packages/omnibias-core/src/omnibias/core",
             REPO_ROOT / "packages/omnibias-torch/src/omnibias/torch",
             REPO_ROOT / "packages/omnibias-keras/src/omnibias/keras",
             REPO_ROOT / "packages/omnibias-jax/src/omnibias/jax"]
    files = {REPO_ROOT / "benchmarks" / name for name in (
        "operator_compiler.py", "operator_report.py", "_operator_models.py", "jet_compiler.py", "_common.py")}
    for root in roots:
        files.update(root.rglob("*.py"))
    return sorted(str(path.relative_to(REPO_ROOT)) for path in files)


def snapshots(paths: list[str]) -> dict[str, Any]:
    result = source_provenance(paths)
    directory = Path(os.environ.get("OMNIBIAS_SCRATCH", REPO_ROOT / "artifacts")) / "operator_sources"
    directory.mkdir(parents=True, exist_ok=True)
    stored = {}
    for relative, digest in result["source_sha256"].items():
        content = (REPO_ROOT / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError("source changed while taking benchmark snapshot")
        target = directory / f"{digest}.py"
        if not target.exists():
            target.write_bytes(content)
        stored[relative] = str(target)
    result["source_snapshots"] = stored
    return result


def baseline_snapshot(module, *, package: bool = False) -> dict[str, Any]:
    """Preserve small optional baseline implementations, not whole frameworks."""
    source = Path(inspect.getsourcefile(module))
    paths = sorted(source.parent.rglob("*.py")) if package else [source]
    directory = Path(os.environ.get("OMNIBIAS_SCRATCH", REPO_ROOT / "artifacts")) / "operator_sources"
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for path in paths:
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        target = directory / f"{digest}.py"
        if not target.exists():
            target.write_bytes(content)
        records[str(path)] = {"sha256": digest, "snapshot": str(target)}
    return records


def admissible(family: str, order: int, method: str, nested_max_order: int) -> str | None:
    if method == "keras-jax" and family == "separable":
        return "this Keras baseline adapter currently captures Dense model families"
    if method == "jax-manual" and family != "ridge":
        return "the handwritten JAX contraction adapter covers the ridge query"
    if family in {"ridge", "bottleneck"} and order % 2:
        return "Laplacian powers require even derivative order"
    if family == "bottleneck" and order not in (2, 4):
        return "this transported-operator benchmark covers Laplacian and biharmonic"
    if method in {"nested", "jax-nested", "folx"} and order > nested_max_order:
        return f"nested derivative order exceeds explicit budget {nested_max_order}"
    if method == "jax-jet" and family != "coupled":
        return "JAX univariate jet is compared on the directional coupled-network query"
    if method == "folx" and family not in {"ridge", "bottleneck"}:
        return "folx baseline implements Laplacian powers"
    if method == "torch-jet" and not (family == "coupled" or (family in {"ridge", "bottleneck"} and order in (2, 4))):
        return "public torch-jet baseline covers directional jets and exact Laplacian/biharmonic"
    return None


def check_values(actual: list[float], expected: list[str]) -> dict[str, Any]:
    metrics = [accuracy(got, want) for got, want in zip(actual, expected, strict=True)]
    # Relative comparison is essential for high mixed orders whose values can
    # be tiny. The absolute floor protects only subnormal-scale references.
    for got, want in zip(actual, expected, strict=True):
        if not math.isfinite(got) or abs(got-float(want)) > 2e-9 * abs(float(want)) + 1e-280:
            raise ArithmeticError(f"independent 80-digit reference failed: {got} versus {want}")
    return {"reference_digits": 80, "rtol": 2e-9, "atol": 1e-280, "points": metrics}


def progress(row: dict[str, Any], stage: str, state: str = "completed") -> None:
    """Flush boundaries before expensive work, retaining the active failure stage."""
    event = {"stage": stage, "state": state, "engine_elapsed_seconds": time.perf_counter()-ENGINE_STARTED}
    row.setdefault("stage_events", []).append(event)
    row["active_stage"] = stage if state == "started" else None
    if state == "completed":
        row["completed_stage"] = stage
    print(PROGRESS + json.dumps(row, allow_nan=False), flush=True)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def reference_request(data: dict[str, Any], order: int) -> dict[str, Any]:
    """Identity includes every binary64 operand, the operator, and oracle revision."""
    return {"schema": REFERENCE_SCHEMA, "operands": data, "order": order,
            "indices": sorted({0, len(data["points"])//2, len(data["points"])-1}),
            "digits": 80, "dtype": "float64", "direction": "all ones" if data["family"] == "coupled" else None,
            "oracle_source_sha256": hashlib.sha256((REPO_ROOT / "benchmarks/_operator_models.py").read_bytes()).hexdigest(),
            "mpmath_version": importlib.metadata.version("mpmath")}


def prepare_reference(data: dict[str, Any], order: int) -> dict[str, Any]:
    """Run the independent oracle once, outside the bounded engine processes."""
    request = reference_request(data, order)
    start = time.perf_counter()
    expected = high_precision(data, order, request["indices"])
    elapsed = time.perf_counter()-start
    if request != reference_request(data, order):
        raise RuntimeError("oracle source changed during reference generation")
    return {"request": request, "key": _json_digest(request), "expected": expected,
            "values_sha256": _json_digest(expected), "generation_seconds": elapsed}


def validate_reference(reference: dict[str, Any], data: dict[str, Any], order: int) -> None:
    request = reference_request(data, order)
    if reference.get("key") != _json_digest(request) or _json_digest(reference.get("request")) != _json_digest(request):
        raise ValueError("reference does not match exact operands, request, or oracle revision")
    values = reference.get("expected")
    if not isinstance(values, list) or len(values) != len(request["indices"]) or reference.get("values_sha256") != _json_digest(values):
        raise ValueError("reference values are missing or their checksum changed")
    if not all(isinstance(value, str) and math.isfinite(float(value)) for value in values):
        raise ValueError("reference requires finite high-precision decimal values")
    if not isinstance(reference.get("generation_seconds"), (float, int)) or not math.isfinite(reference["generation_seconds"]) or reference["generation_seconds"] < 0:
        raise ValueError("reference generation cost must be finite and nonnegative")


def memory_summary(events: list[list[int]]) -> dict[str, int]:
    """Sum signed raw profiler events; version changes have paired +/- events."""
    baseline = sum(size for _, action, size, _ in events if action == 1)
    current, peak = 0, 0
    for _, _, size, _ in events:
        current += size
        peak = max(peak, current)
    return {"preexisting_bytes": baseline, "peak_live_bytes": peak,
            "peak_above_preexisting_bytes": max(0, peak-baseline)}


def torch_memory(action) -> dict[str, Any]:
    """One untimed pass; record tracked CPU tensor storage separately from RSS."""
    import torch
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    profile_memory=True, record_shapes=True, with_stack=True) as profile:
            retained = action()
        with tempfile.TemporaryDirectory(prefix="omnibias-memory-") as directory:
            target = str(Path(directory)/"memory.raw.json.gz")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*export_memory_timeline.*")
                profile.export_memory_timeline(target)
            with gzip.open(target, "rt") as stream:
                result = memory_summary(json.load(stream))
        del retained
        return result | {"status": "ok", "scope": "PyTorch CPU profiler tracked live tensor storage; untimed warm pass; excludes untracked allocator/workspace memory"}
    except (RuntimeError, ValueError, AssertionError) as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _torch(args: argparse.Namespace, data: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    progress(row, "imports", "started")
    import torch
    from omnibias.core.jet_algebra import JetSpec
    from omnibias.core.jet_series import activation
    from omnibias.torch.experimental import OperatorSpec, compile_operator
    from omnibias.torch.experimental.compiler import _OPS
    progress(row, "imports")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = torch_model(data)
    x = torch.tensor(data["points"], dtype=torch.float64)
    parameters = tuple(model.parameters())
    initial = tuple(p.detach().clone() for p in parameters)
    family, order = args.family, args.order
    row["backend_version"] = torch.__version__
    request = (OperatorSpec.directional(order) if family == "coupled" else
               OperatorSpec.mixed((1,)*order) if family == "separable" else
               OperatorSpec.laplacian(power=order//2))
    progress(row, "capture", "started")
    start = time.perf_counter()
    if args.worker in {"compiler", "reference"}:
        program = compile_operator(model, (x,), operator=request, batch_ndim=1,
                                   activation_method=args.activation_method,
                                   kernel_lowering=args.kernel_lowering,
                                   strategy="reference" if args.worker == "reference" else "auto")
        row["plan"] = program.explain()
        operation = (lambda z: program(z, direction=torch.ones_like(z))) if family == "coupled" else program
    elif args.worker == "manual":
        if family == "ridge":
            spec = JetSpec.directional(order)
            def operation(z):
                w, b, c, _ = tuple(model.parameters())
                pre = torch.nn.functional.linear(z, w, b)
                if args.activation_method == "polynomial":
                    from omnibias.torch.fastpath.legendre import tanh_nth_derivative
                    derivative = tanh_nth_derivative(pre, order)
                else:
                    derivative = activation([pre, torch.ones_like(pre)] + [torch.zeros_like(pre)]*(order-1),
                                            spec, "tanh", _OPS)[order] * float(math.factorial(order))
                return (derivative * w.square().sum(-1).pow(order//2)) @ c.T
        elif family == "separable":
            def operation(z):
                w, b = tuple(model.parameters())
                inner = torch.tanh(z*w+b)
                total = inner.sum(-1, keepdim=True)
                outer = (torch.sin(total), torch.cos(total), -torch.sin(total), -torch.cos(total))[order % 4]
                return outer * (w*(1-inner.square())).prod(-1, keepdim=True)
        elif family == "coupled":
            from omnibias.torch.jet import mlp_jet
            def operation(z):
                params = tuple(model.parameters())
                layers = [(params[2*i], params[2*i+1], "tanh" if act else None)
                          for i, (_, _, act) in enumerate(data["layers"])]
                return mlp_jet(z, torch.ones_like(z), layers, order, riccati=True)[order] * float(math.factorial(order))
        else:
            from omnibias.torch.experimental._operator_metric import (
                laplacian_activation,
                laplacian_affine,
                laplacian_seed,
                metric_activation,
                metric_affine,
                metric_seed,
            )
            def operation(z):
                params = tuple(model.parameters())
                basis, bias = params[:2]
                gram = basis @ basis.T
                latent = torch.nn.functional.linear(z, basis, bias)
                if order == 2:
                    state = laplacian_activation(laplacian_seed(latent, args.latent), gram)
                    for i, (_, _, act) in enumerate(data["layers"][1:], 1):
                        state = laplacian_affine(state, params[2*i], params[2*i+1])
                        if act:
                            state = laplacian_activation(state, gram)
                    return state.laplacian
                fourth = metric_activation(metric_seed(latent, args.latent), gram, args.latent)
                for i, (_, _, act) in enumerate(data["layers"][1:], 1):
                    fourth = metric_affine(fourth, params[2*i], params[2*i+1])
                    if act:
                        fourth = metric_activation(fourth, gram, args.latent)
                return fourth.biharmonic
    elif args.worker == "torch-jet":
        import jet as jet_package
        from jet import bilaplacian, capture_graph, common_subexpression_elimination, jet, laplacian
        row["baseline_version"] = importlib.metadata.version("jet-for-pytorch")
        row["baseline_sources"] = baseline_snapshot(jet_package, package=True)
        if family == "coupled":
            transformed = jet(model, (x,))
            def operation(z):
                return transformed((z, torch.ones_like(z), *([torch.zeros_like(z)]*(order-1))))[order]
        else:
            transform = laplacian if order == 2 else bilaplacian
            transformed = transform(model, (x[0],), collapsed=True, randomization=None)
            operation = torch.func.vmap(transformed)
        source = inspect.getsourcefile(transformed)
        row["baseline_source_path"] = source
        if args.execution == "compiled":
            # The public transform returns an FX interpreter. The package's
            # documented compile path first freezes it via capture_graph.
            # Seeding is inside operation, exposing constant zeros equally.
            captured, _ = capture_graph(operation, (x,))
            common_subexpression_elimination(captured.graph)
            captured.recompile()
            operation = captured
            row["baseline_lowering"] = "public capture_graph plus common_subexpression_elimination before torch.compile"
    else:
        def point(z):
            return model(z).squeeze()
        function = point
        if family in {"ridge", "bottleneck"}:
            for _ in range(order//2):
                previous = function
                def function(z, f=previous):
                    return torch.func.hessian(f)(z).diagonal().sum()
        else:
            for i in range(order):
                previous = function
                if family == "separable":
                    direction = torch.eye(data["dimension"], dtype=x.dtype)[i]
                    def function(z, f=previous, v=direction):
                        return torch.func.jvp(f, (z,), (v,))[1]
                else:
                    def function(z, f=previous):
                        return torch.func.jvp(f, (z,), (torch.ones_like(z),))[1]
        def operation(z):
            return torch.func.vmap(function)(z).unsqueeze(-1)
    row["capture_ms"] = (time.perf_counter()-start)*1000
    progress(row, "capture")
    progress(row, "compile_setup", "started")
    start = time.perf_counter()
    if args.execution == "compiled":
        if isinstance(operation, torch.nn.Module):
            # Use the standard in-place entrypoint for every module baseline.
            # It preserves hooks without adding an outer OptimizedModule call.
            operation.compile(fullgraph=True)
            row["compilation_entrypoint"] = "nn.Module.compile(fullgraph=True)"
        else:
            operation = torch.compile(operation, fullgraph=True)
            row["compilation_entrypoint"] = "torch.compile(function, fullgraph=True)"
    row["compile_setup_ms"] = (time.perf_counter()-start)*1000
    row["compilation_scope"] = "Torch compilation is lazy; compile_and_first_execution includes actual forward compilation; backward may compile separately"
    progress(row, "compile_setup")
    progress(row, "compile_and_first_execution", "started")
    start = time.perf_counter()
    result = operation(x)
    row["compile_and_first_execution_ms"] = (time.perf_counter()-start)*1000
    progress(row, "compile_and_first_execution")
    progress(row, "accuracy", "started")
    indices = row["reference"]["request"]["indices"]
    row["accuracy"] = check_values(result.detach().reshape(-1)[indices].tolist(), row["reference"]["expected"])
    progress(row, "accuracy")
    progress(row, "warm_forward", "started")
    row["evaluation"] = timing(lambda: None, lambda _: operation(x), repeats=args.repeats, warmup=args.warmup)
    progress(row, "warm_forward")
    # Check derivative dependence, not merely a primal term in a combined loss.
    progress(row, "gradient_validation", "started")
    start = time.perf_counter()
    grads = torch.autograd.grad(operation(x).square().mean(), parameters, allow_unused=True)
    row["gradient_validation_first_call_ms"] = (time.perf_counter()-start)*1000
    row["gradient_norms"] = [None if g is None else float(g.norm()) for g in grads]
    if any(g is not None and not bool(torch.isfinite(g).all()) for g in grads):
        raise ArithmeticError("nonfinite derivative-only parameter gradient")
    if not any(g is not None and bool(torch.isfinite(g).all()) and bool(g.abs().max() > 0) for g in grads):
        raise ArithmeticError("no finite nonzero derivative-only parameter gradient")
    progress(row, "gradient_validation")
    if args.forward_only:
        progress(row, "tensor_memory", "started")
        row["tensor_memory"] = torch_memory(lambda: operation(x))
        progress(row, "tensor_memory")
        return row | {"status": "ok", "training": "not requested"}
    optimizer = torch.optim.SGD(parameters, lr=1e-4)
    def reset():
        with torch.no_grad():
            for p, value in zip(parameters, initial, strict=True):
                p.copy_(value)
        optimizer.zero_grad(set_to_none=True)
    def loss():
        return operation(x).square().mean()
    def prepare():
        reset()
        return loss()
    progress(row, "backward", "started")
    row["backward"] = timing(prepare, lambda objective: objective.backward(), repeats=args.repeats, warmup=args.warmup)
    progress(row, "backward")
    def train(_):
        objective = loss()
        objective.backward()
        optimizer.step()
        return objective
    progress(row, "optimizer", "started")
    row["training_step"] = timing(reset, train, repeats=args.repeats, warmup=args.warmup)
    row["training_scope"] = "compiled derivative forward (when requested), Python loss/backward/SGD orchestration"
    row["original_parameters_updated"] = any(not torch.equal(p, value) for p, value in zip(parameters, initial, strict=True))
    if not row["original_parameters_updated"]:
        raise ArithmeticError("derivative-only optimizer step did not update original parameters")
    progress(row, "optimizer")
    progress(row, "tensor_memory", "started")
    reset()
    saved = {}
    def pack(tensor):
        storage = tensor.untyped_storage()
        saved[storage.data_ptr()] = storage.nbytes()
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss().backward()
    row["saved_tensor_storage_bytes"] = sum(saved.values())
    row["saved_tensor_scope"] = "unique saved storage across one forward/backward, including parameters/inputs; not allocator peak"
    reset()
    row["tensor_memory"] = torch_memory(lambda: train(None))
    progress(row, "tensor_memory")
    return row | {"status": "ok"}


def _jax(args: argparse.Namespace, data: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    progress(row, "imports", "started")
    import jax
    import jax.numpy as jnp
    if args.worker == "keras-jax":
        os.environ["KERAS_BACKEND"] = "jax"
        import keras
        from omnibias.core.operator_spec import OperatorSpec
        from omnibias.keras.experimental import operator_model
    elif args.worker == "jax-jet":
        import jax.experimental.jet
    elif args.worker == "folx":
        import folx
    progress(row, "imports")
    jax.config.update("jax_enable_x64", True)
    progress(row, "capture", "started")
    x = jnp.asarray(data["points"])
    parameters = (tuple(map(jnp.asarray, (data["weights"], data["biases"]))) if args.family == "separable" else
                  tuple(jnp.asarray(v) for w, b, _ in data["layers"] for v in (w, b)))
    def primal(params, coordinate):
        if args.family == "separable":
            return jnp.sin(jnp.tanh(coordinate*params[0]+params[1]).sum(-1))
        for i, (_, _, act) in enumerate(data["layers"]):
            coordinate = coordinate @ params[2*i].T + params[2*i+1]
            if act:
                coordinate = jnp.tanh(coordinate)
        return coordinate.squeeze(-1)
    def evaluate(params, coordinate):
        def function(point):
            return primal(params, point)
        if args.worker == "jax-jet":
            from jax.experimental.jet import jet
            return jet(function, (coordinate,), ((jnp.ones_like(coordinate),) + (jnp.zeros_like(coordinate),)*(args.order-1),))[1][-1]
        if args.family in {"ridge", "bottleneck"}:
            for _ in range(args.order//2):
                previous = function
                if args.worker == "folx":
                    import folx
                    transformed = folx.forward_laplacian(previous)
                    def function(point, f=transformed):
                        return f(point).laplacian
                else:
                    def function(point, f=previous):
                        return jnp.trace(jax.hessian(f)(point))
        else:
            for i in range(args.order):
                previous = function
                direction = jnp.eye(data["dimension"])[i] if args.family == "separable" else jnp.ones(data["dimension"])
                def function(point, f=previous, v=direction):
                    return jax.jvp(f, (point,), (v,))[1]
        return jax.vmap(function)(coordinate)
    row["backend_version"] = jax.__version__
    if args.worker == "keras-jax":
        start = time.perf_counter()
        inputs = keras.Input((data["dimension"],), dtype="float64")
        state = inputs
        for weight, _, act in data["layers"]:
            state = keras.layers.Dense(len(weight), activation="tanh" if act else None, dtype="float64")(state)
        source = keras.Model(inputs, state)
        source.set_weights([p.T if i % 2 == 0 else p for i, p in enumerate(parameters)])
        request = OperatorSpec.directional(args.order) if args.family == "coupled" else OperatorSpec.laplacian(args.order//2)
        program = operator_model(source, operator=request, activation_method=args.activation_method,
                                 precision=args.precision)
        program([x, jnp.ones_like(x)] if args.family == "coupled" else x)
        row["frontend_capture_ms"] = (time.perf_counter()-start)*1000
        row["frontend_capture_scope"] = "Keras graph transformation plus one ordinary call to build the wrapper before stateless JIT"
        row["frontend_version"] = keras.__version__
        row["plan"] = program.explain()
        def evaluate(params, coordinate):
            variables = [p.T if i % 2 == 0 else p for i, p in enumerate(params)]
            arguments = [coordinate, jnp.ones_like(coordinate)] if args.family == "coupled" else coordinate
            result, _ = program.stateless_call(variables, [], arguments, training=False)
            return result.squeeze(-1)
    elif args.worker == "jax-manual":
        from types import SimpleNamespace

        from omnibias.core.jet_algebra import JetSpec
        from omnibias.core.jet_series import activation
        spec = JetSpec.directional(args.order)
        ops = SimpleNamespace(zeros_like=jnp.zeros_like, ones_like=jnp.ones_like,
                              tanh=jnp.tanh, sigmoid=jax.nn.sigmoid)
        def evaluate(params, coordinate):
            weight, bias, readout, _ = params
            value = coordinate @ weight.T + bias
            if args.activation_method == "polynomial":
                from omnibias.jax._fastpath import tanh_nth_derivative
                derivative = tanh_nth_derivative(value, args.order)
            else:
                derivative = activation([value, jnp.ones_like(value)] + [jnp.zeros_like(value)]*(args.order-1),
                                        spec, "tanh", ops)[args.order] * float(math.factorial(args.order))
            return ((derivative * jnp.sum(weight**2, axis=-1)**(args.order//2)) @ readout.T).squeeze(-1)
    if args.worker == "jax-jet":
        import jax.experimental.jet as jet_module
        row["baseline_sources"] = baseline_snapshot(jet_module)
    elif args.worker == "folx":
        import folx
        row["baseline_version"] = importlib.metadata.version("folx")
        row["baseline_sources"] = baseline_snapshot(folx, package=True)
    operation = jax.jit(evaluate) if args.execution == "compiled" else evaluate
    start = time.perf_counter()
    if args.execution == "compiled":
        lowered = operation.lower(parameters, x)
        row["capture_ms"] = (time.perf_counter()-start)*1000
        progress(row, "capture")
        progress(row, "compilation", "started")
        start = time.perf_counter()
        operation = lowered.compile()
        row["compilation_ms"] = (time.perf_counter()-start)*1000
        progress(row, "compilation")
        estimate = operation.memory_analysis()
        if estimate is not None:
            row["compiled_buffer_estimate"] = {name: getattr(estimate, name) for name in
                ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")}
            row["compiled_buffer_scope"] = "XLA static buffer estimate, not measured peak allocation"
    else:
        progress(row, "capture")
    progress(row, "first_execution", "started")
    start = time.perf_counter()
    values = operation(parameters, x)
    values.block_until_ready()
    row["first_execution_ms"] = (time.perf_counter()-start)*1000
    progress(row, "first_execution")
    progress(row, "accuracy", "started")
    indices = row["reference"]["request"]["indices"]
    row["accuracy"] = check_values([float(values[i]) for i in indices], row["reference"]["expected"])
    progress(row, "accuracy")
    progress(row, "warm_forward", "started")
    row["evaluation"] = timing(lambda: None, lambda _: operation(parameters, x), repeats=args.repeats, warmup=args.warmup)
    progress(row, "warm_forward")
    progress(row, "gradient_validation", "started")
    gradient_function = jax.grad(lambda params, coordinate: jnp.mean(evaluate(params, coordinate)**2))
    if args.execution == "compiled":
        gradient_function = jax.jit(gradient_function)
    start = time.perf_counter()
    grads = gradient_function(parameters, x)
    jax.block_until_ready(grads)
    row["gradient_validation_first_call_ms"] = (time.perf_counter()-start)*1000
    row["gradient_norms"] = [float(jnp.linalg.norm(g)) for g in grads]
    if not all(bool(jnp.isfinite(g).all()) for g in grads) or not any(bool(jnp.any(g != 0)) for g in grads):
        raise ArithmeticError("no finite nonzero derivative-only parameter gradient")
    progress(row, "gradient_validation")
    if not args.forward_only:
        def train(params, coordinate):
            objective, gradients = jax.value_and_grad(lambda p: jnp.mean(evaluate(p, coordinate)**2))(params)
            return objective, jax.tree.map(lambda p, g: p-1e-4*g, params, gradients), gradients
        training = jax.jit(train) if args.execution == "compiled" else train
        progress(row, "optimizer", "started")
        row["training_step"] = timing(lambda: None, lambda _: training(parameters, x), repeats=args.repeats, warmup=args.warmup)
        row["training_scope"] = "whole functional loss/gradient/SGD update compiled together when requested"
        updated = training(parameters, x)[1]
        row["functional_parameters_updated"] = any(not bool(jnp.array_equal(p, v)) for p, v in zip(parameters, updated, strict=True))
        if not row["functional_parameters_updated"]:
            raise ArithmeticError("derivative-only functional SGD did not update parameters")
        progress(row, "optimizer")
    row["backward"] = {"status": "unavailable", "reason": "measured as part of functional training step"}
    return row | {"status": "ok"}


def worker(args: argparse.Namespace) -> dict[str, Any]:
    _configure_process(args.threads)
    row = {"family": args.family, "order": args.order, "method": args.worker,
           "execution": args.execution, "activation_method": args.activation_method,
           "protocol": SCHEMA, "cache_policy": args.cache, "kernel_lowering": args.kernel_lowering,
           "precision": args.precision, "cache_protocol": cache_protocol(args.cache)}
    row["compiler_cache_environment"] = {key: os.environ.get(key) for key in (
        "TORCHINDUCTOR_CACHE_DIR", "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
        "TORCHINDUCTOR_COMPILE_THREADS", "JAX_COMPILATION_CACHE_DIR",
        "JAX_ENABLE_COMPILATION_CACHE", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS")}
    progress(row, "preparation", "started")
    reason = admissible(args.family, args.order, args.worker, args.nested_max_order)
    if reason:
        return row | {"status": "refused", "reason": reason}
    paths = source_paths()
    sources = snapshots(paths)
    row["source_provenance"] = sources
    data = model_data(args.family, args.order, args.batch, args.dim, args.hidden, args.latent, args.seed)
    row["operands"] = data
    try:
        if args.reference_file is None:
            raise ValueError("engine worker requires a parent-precomputed --reference-file")
        reference = json.loads(args.reference_file.read_text())
        validate_reference(reference, data, args.order)
        row["reference"] = reference
        row["reference_generation_seconds"] = reference["generation_seconds"]
        progress(row, "preparation")
        runner = _jax if args.worker in {"jax-jet", "jax-nested", "folx", "keras-jax", "jax-manual"} else _torch
        result = runner(args, data, row)
    except ImportError as exc:
        result = row | {"status": "unavailable", "reason": str(exc)}
    except Exception as exc:
        refused = isinstance(exc, ValueError) and "budget" in str(exc).lower()
        result = row | {"status": "refused" if refused else "error", "reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-5000:]}
    result["peak_process_rss_mib"] = rss_mb()
    result["source_changed_during_cell"] = sources["source_sha256"] != source_provenance(paths)["source_sha256"]
    result["rss_scope"] = "engine worker high-water RSS including imports and compilation, excluding separately generated oracle; not tensor allocation"
    result["cpu_affinity"] = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return result


def worker_command(args: argparse.Namespace, family: str, order: int, method: str) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", method,
               "--family", family, "--order", str(order)]
    for name in ("execution", "activation_method", "kernel_lowering", "precision", "repeats", "warmup", "batch", "dim", "hidden",
                 "latent", "seed", "threads", "nested_max_order", "cache"):
        command.extend(["--"+name.replace("_", "-"), str(getattr(args, name))])
    if args.forward_only:
        command.append("--forward-only")
    return command


def cache_protocol(policy: str) -> dict[str, Any]:
    """Describe reuse independently of random per-process cache directory paths."""
    return {
        "version": 2,
        "initial_state": "shared persistent cache" if policy == "persistent" else "empty isolated cache directories",
        "within_worker_reuse": "normal backend reuse" if policy != "disabled" else "diagnostic backend-specific disabling",
        "torch_force_disable_caches": policy == "disabled",
        "jax_persistent_compilation_cache": policy != "disabled",
        "scope": "fresh process for every cell; disabled uses Torch force-disable and JAX persistent-cache controls, whose cache layers differ",
    }


def cache_environment(policy: str, isolated_root: Path) -> dict[str, str]:
    """A cold run starts empty; it must not suppress reuse during that run."""
    environment = dict(os.environ)
    if policy in {"cold", "disabled"}:
        environment.update(TORCHINDUCTOR_CACHE_DIR=str(isolated_root / "inductor"),
                           JAX_COMPILATION_CACHE_DIR=str(isolated_root / "jax"))
    else:
        cache = Path(os.environ.get("OMNIBIAS_SCRATCH", REPO_ROOT / "artifacts")) / "operator_cache"
        environment.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / "inductor"))
        environment.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache / "jax"))
    # CLI cache policy is explicit, including when the parent inherited a
    # diagnostic disable flag from a previous experiment.
    environment.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
    if policy == "disabled":
        environment["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    environment["JAX_ENABLE_COMPILATION_CACHE"] = "false" if policy == "disabled" else "true"
    environment.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    return environment


def run_case(args: argparse.Namespace, family: str, order: int, method: str,
             reference: dict[str, Any] | None = None) -> dict[str, Any]:
    reason = admissible(family, order, method, args.nested_max_order)
    if reason:
        return {"family": family, "order": order, "method": method, "protocol": SCHEMA,
                "status": "refused", "reason": reason}
    data = model_data(family, order, args.batch, args.dim, args.hidden, args.latent, args.seed)
    reference = prepare_reference(data, order) if reference is None else reference
    validate_reference(reference, data, order)
    with tempfile.TemporaryDirectory(prefix="omnibias-operator-") as directory:
        root = Path(directory)
        reference_path = root / "reference.json"
        reference_path.write_text(json.dumps(reference, allow_nan=False))
        environment = cache_environment(args.cache, root)
        with (root/"stdout").open("w+") as stdout, (root/"stderr").open("w+") as stderr:
            start = time.monotonic()
            command = worker_command(args, family, order, method) + ["--reference-file", str(reference_path)]
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                       env=environment, start_new_session=(os.name == "posix"))
            failure, peak = None, 0.
            while process.poll() is None:
                try:
                    status = Path(f"/proc/{process.pid}/status").read_text()
                    memory = next(float(line.split()[1])/1024 for line in status.splitlines() if line.startswith("VmRSS:"))
                    peak = max(peak, memory)
                except (OSError, StopIteration):
                    pass
                if time.monotonic()-start > args.timeout:
                    failure = "timeout"
                elif peak > args.rss_limit_mib:
                    failure = "memory_budget"
                if failure:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    break
                time.sleep(.05)
            process.wait()
            stdout.seek(0)
            stderr.seek(0)
            complete, partial = {}, {}
            for line in stdout.read().splitlines():
                if line.startswith(PREFIX):
                    complete = json.loads(line[len(PREFIX):])
                elif line.startswith(PROGRESS):
                    partial = json.loads(line[len(PROGRESS):])
            result = complete or partial or {"family": family, "order": order, "method": method}
            if failure:
                result.update(status=failure, reason="parent terminated worker at configured resource budget")
            elif process.returncode or not complete:
                result.update(status="error", reason="worker exited without a complete structured result")
            engine_seconds = time.monotonic()-start
            result.update(isolated_wall_seconds=engine_seconds, engine_wall_seconds=engine_seconds,
                          observed_worker_rss_mib=peak, protocol=SCHEMA,
                          cache_protocol=cache_protocol(args.cache),
                          reference=reference, reference_generation_seconds=reference["generation_seconds"],
                          end_to_end_seconds=reference["generation_seconds"]+engine_seconds,
                          end_to_end_scope="one independent reference generation plus this engine worker; reference reused across report cells")
            errors = stderr.read()
            if errors:
                result["stderr_tail"] = errors[-2000:]
            return result


def query_signature(row: dict[str, Any]) -> str:
    return _json_digest({name: row.get(name) for name in (
        "protocol", "execution", "activation_method", "operands", "cache_policy", "cache_protocol", "kernel_lowering", "precision")})


def reference_evidence_valid(row: dict[str, Any]) -> bool:
    """Validate recorded v2 evidence against its snapshot, never today's oracle."""
    if row.get("protocol") != SCHEMA:
        return True  # v1 ran the oracle inside each worker and has no separate record.
    try:
        reference, data = row["reference"], row["operands"]
        request, values = reference["request"], reference["expected"]
        return bool(
            request["schema"] == REFERENCE_SCHEMA and request["digits"] == 80 and request["dtype"] == "float64"
            and request["order"] == row["order"] and data["family"] == row["family"]
            and _json_digest(request["operands"]) == _json_digest(data)
            and request["indices"] == sorted({0, len(data["points"])//2, len(data["points"])-1})
            and request["direction"] == ("all ones" if row["family"] == "coupled" else None)
            and reference["key"] == _json_digest(request) and reference["values_sha256"] == _json_digest(values)
            and request["oracle_source_sha256"] == row["source_provenance"]["source_sha256"]["benchmarks/_operator_models.py"]
            and len(values) == len(request["indices"]) and all(isinstance(v, str) and math.isfinite(float(v)) for v in values)
            and math.isfinite(reference["generation_seconds"]) and reference["generation_seconds"] >= 0
            and len(row["accuracy"]["points"]) == len(values)
            and all(point["status"] == "ok" for point in row["accuracy"]["points"])
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only complete 3x20 accuracy-gated, unchanged-source cells earn ratios."""
    groups: dict[tuple[str, int, str], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        key = row["family"], row["order"], query_signature(row)
        groups.setdefault(key, {}).setdefault(row["method"], []).append(row)
    result = []
    for (family, order, query_hash), methods in sorted(groups.items()):
        measured = {}
        sources = {}
        for method, cells in methods.items():
            if len(cells) < 3 or any(cell.get("status") != "ok" or cell.get("source_changed_during_cell", True)
                                    or not reference_evidence_valid(cell)
                                    or len(cell.get("evaluation", {}).get("samples_ms", [])) < 20 for cell in cells):
                continue
            fingerprints = {json.dumps(cell["source_provenance"]["source_sha256"], sort_keys=True) for cell in cells}
            if len(fingerprints) != 1:
                continue
            versions = {tuple(cell.get(key) for key in ("backend_version", "frontend_version", "baseline_version")) for cell in cells}
            if len(versions) != 1 or len({cell.get("replicate") for cell in cells}) < 3:
                continue
            measured[method] = statistics.median(cell["evaluation"]["median_ms"] for cell in cells)
            sources[method] = next(iter(fingerprints))
        for compiled_method, handwritten in (("compiler", "manual"), ("keras-jax", "jax-manual")):
            if compiled_method not in measured:
                continue
            comparable = {method: value for method, value in measured.items() if sources[method] == sources[compiled_method]}
            compiler = comparable[compiled_method]
            row = {"family": family, "order": order, "compiled_method": compiled_method,
                   "query_configuration_sha256": query_hash,
                   "median_of_process_medians_ms": comparable,
                   "baseline_over_compiler": {name: value/compiler for name, value in comparable.items() if name != compiled_method}}
            baselines = {name: value for name, value in comparable.items() if name != compiled_method}
            if baselines:
                fastest = min(baselines, key=baselines.get)
                row.update(fastest_measured_baseline=fastest, fastest_baseline_over_compiler=baselines[fastest]/compiler)
            if family == "ridge" and handwritten in comparable:
                row["ridge_release_gate_pass"] = compiler <= comparable[handwritten] + max(.005, .2*comparable[handwritten])
            result.append(row)
    return result


def main() -> None:
    args = parser().parse_args()
    validate(args)
    if args.worker:
        if args.family is None or args.order is None:
            raise ValueError("worker requires family and order")
        print(PREFIX+json.dumps(worker(args), allow_nan=False), flush=True)
        return
    cases = [(family, order, method, replicate) for replicate in range(args.replicates)
             for family in args.families for order in args.orders for method in args.methods]
    random.Random(args.seed).shuffle(cases)
    started = time.perf_counter()
    references = {}
    for family, order in sorted({(family, order) for family, order, method, _ in cases
                                if admissible(family, order, method, args.nested_max_order) is None}):
        print(f"reference generation started: {family} order={order}", flush=True)
        data = model_data(family, order, args.batch, args.dim, args.hidden, args.latent, args.seed)
        references[family, order] = prepare_reference(data, order)
        print(f"reference generation completed: {family} order={order}", flush=True)
    rows = []
    for family, order, method, replicate in cases:
        row = run_case(args, family, order, method, references.get((family, order))) | {"replicate": replicate}
        rows.append(row)
        print(f"{family} order={order} {method} replicate={replicate}: {row['status']}", flush=True)
        payload = {"schema": SCHEMA, "config": vars(args), "rows": rows,
                   "scope": "float64 CPU; derivative-only forward, backward and SGD; runtime coordinates/parameters",
                   "source_snapshot_directory": "operator_sources", "cache_policy": args.cache,
                   "cache_protocol": cache_protocol(args.cache),
                   "references": {value["key"]: value for value in references.values()},
                   "reference_generation_total_seconds": sum(value["generation_seconds"] for value in references.values()),
                   "wall_seconds_through_measurements": time.perf_counter()-started,
                   "resource_budget_scope": "engine worker wall time/RSS including imports, compilation, backward and optimizer; independent oracle generation excluded and recorded separately; RSS excludes compiler children"}
        payload["comparisons"] = comparisons(rows)
        write_json(args.output, payload)
    payload["environment"] = provenance(schema=SCHEMA, config={})
    payload["environment"]["host_hardware"] = hardware_details()
    print(f"wrote {write_json(args.output, payload)}", flush=True)


if __name__ == "__main__":
    main()
