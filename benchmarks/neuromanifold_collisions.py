# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Reproducible ordinary-pair versus confluent-coordinate numerical experiment.

Run without flags for a small smoke experiment; --full20 uses twenty fixed
seeds with identical sample/parameter/optimizer-step budgets in both charts.
Results describe this finite workload, not generic convergence or discovery.
Python allocation peaks are reported separately from explicit tensor state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import tracemalloc
from collections.abc import Callable
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from omnibias.torch.confluence import collision_atom, retract_spread
from omnibias.torch.fastpath.eulerian import sigmoid_nth_derivative
from torch import Tensor


def ordinary_pair(x: Tensor, parameters: Tensor) -> Tensor:
    """Coordinates (a_plus, a_minus, b_plus, b_minus)."""
    return parameters[0] * torch.sigmoid(x + parameters[2]) + parameters[1] * torch.sigmoid(x + parameters[3])


def confluent_pair(x: Tensor, parameters: Tensor) -> Tensor:
    """Coordinates (m0, m1, mean, rho), including attained rho=0."""
    return collision_atom(x + parameters[2], parameters[3], torch.ones_like(x), parameters[0], parameters[1])


def ordinary_coordinates(moment_coordinates: Tensor) -> Tensor:
    m0, m1, mean, rho = moment_coordinates
    half = torch.sqrt(rho)
    return torch.stack((0.5 * (m0 + m1 / half), 0.5 * (m0 - m1 / half), mean + half, mean - half))


def condition_number(fn: Callable[[Tensor], Tensor], parameters: Tensor) -> dict[str, float | int | None]:
    jacobian = torch.func.jacfwd(fn)(parameters)
    singular = torch.linalg.svdvals(jacobian)
    condition = float(singular[0] / singular[-1])
    rank = int(torch.linalg.matrix_rank(jacobian))
    return {"condition": condition if math.isfinite(condition) else None,
            "smallest_singular": float(singular[-1]), "numerical_rank": rank}


def conditioning_scan() -> list[dict[str, Any]]:
    x = torch.linspace(-2, 2, 129, dtype=torch.float64)
    rows = []
    for half in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8):
        params = torch.tensor([0.0, 1.0, 0.0, half * half], dtype=torch.float64)
        ordinary = ordinary_coordinates(params)
        with localcontext() as context:
            context.prec = 80
            h = Decimal(float(params[3])).sqrt()
            reference_values = []
            for value in x.tolist():
                coordinate = Decimal(value)
                plus = 1 / (1 + (-(coordinate + h)).exp())
                minus = 1 / (1 + (-(coordinate - h)).exp())
                reference_values.append(float((plus - minus) / (2 * h)))
        reference = torch.tensor(reference_values, dtype=torch.float64)
        rows.append({"half_spread": half,
                     "ordinary": condition_number(lambda p: ordinary_pair(x, p), ordinary),
                     "confluent": condition_number(lambda p: confluent_pair(x, p), params),
                     "ordinary_linf_to_decimal_reference": float((ordinary_pair(x, ordinary) - reference).abs().max()),
                     "confluent_linf_to_decimal_reference": float((confluent_pair(x, params) - reference).abs().max()),
                     "realization_difference_linf": float((ordinary_pair(x, ordinary) - confluent_pair(x, params)).abs().max())})
    return rows


def _tensor_bytes(parameters: Tensor, optimizer: torch.optim.Optimizer) -> int:
    tensors = [parameters]
    if parameters.grad is not None:
        tensors.append(parameters.grad)
    tensors.extend(value for state in optimizer.state.values() for value in state.values() if isinstance(value, Tensor))
    return sum(value.numel() * value.element_size() for value in tensors)


def train_chart(seed: int, chart: str, *, steps: int, samples: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    jitter = torch.randn(3, generator=generator, dtype=torch.float64)
    initial = torch.tensor([0.0, 1.0, 0.0, 0.005], dtype=torch.float64)
    initial[:3] += jitter * torch.tensor([0.01, 0.03, 0.02], dtype=torch.float64)
    start = initial if chart == "confluent" else ordinary_coordinates(initial)
    parameters = start.clone().requires_grad_()
    x = torch.linspace(-2, 2, samples, dtype=torch.float64)
    target = sigmoid_nth_derivative(x, 1)
    fn = confluent_pair if chart == "confluent" else ordinary_pair
    optimizer = torch.optim.Adam([parameters], lr=0.01)
    native = torch.sigmoid
    calls = 0
    boundary_step = None

    def counted(value: Tensor) -> Tensor:
        nonlocal calls
        calls += 1
        return native(value)

    tracemalloc.start()
    start_time = time.perf_counter()
    with patch.object(torch, "sigmoid", counted):
        initial_loss = float(((fn(x, parameters) - target) ** 2).mean().detach())
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = ((fn(x, parameters) - target) ** 2).mean()
            parameters.grad = torch.autograd.grad(loss, parameters)[0]
            optimizer.step()
            if chart == "confluent":
                with torch.no_grad():
                    parameters[3].copy_(retract_spread(parameters[3]))
                if float(parameters[3].detach()) == 0 and boundary_step is None:
                    boundary_step = step + 1
        prediction = fn(x, parameters)
    elapsed = time.perf_counter() - start_time
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    error = prediction.detach() - target
    return {"seed": seed, "chart": chart, "steps": steps, "samples": samples, "trainable_parameters": 4,
            "initial_mse": initial_loss, "final_mse": float((error ** 2).mean()),
            "final_linf": float(error.abs().max()), "seconds": elapsed,
            "sigmoid_calls": calls, "python_peak_bytes": python_peak,
            "parameter_gradient_optimizer_bytes": _tensor_bytes(parameters, optimizer),
            "first_attained_boundary_step": boundary_step,
            "final_rho": float(parameters[3].detach()) if chart == "confluent" else None}


def attained_boundary_demo() -> dict[str, Any]:
    """A live projected training step reaches the derivative atom exactly."""
    x = torch.linspace(-2, 2, 65, dtype=torch.float64)
    rho = torch.tensor(0.005, dtype=torch.float64, requires_grad=True)
    target = sigmoid_nth_derivative(x, 1)
    optimizer = torch.optim.Adam([rho], lr=0.02)
    trace = []
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        value = collision_atom(x, rho, torch.ones_like(x), torch.zeros_like(rho), torch.ones_like(rho))
        loss = ((value - target) ** 2).mean()
        rho.grad = torch.autograd.grad(loss, rho)[0]
        gradient = float(rho.grad) if rho.grad is not None else None
        optimizer.step()
        with torch.no_grad():
            rho.copy_(retract_spread(rho))
        trace.append({"step": step + 1, "rho": float(rho.detach()), "gradient": gradient,
                      "mse_before_step": float(loss.detach())})
    final = collision_atom(x, rho, torch.ones_like(x), torch.zeros_like(rho), torch.ones_like(rho))
    return {"trace": trace, "attained_zero": float(rho.detach()) == 0,
            "derivative_atom_linf": float((final.detach() - target).abs().max())}


def run(*, full20: bool = False) -> dict[str, Any]:
    torch.set_num_threads(1)
    seeds = tuple(range(20)) if full20 else (0, 1)
    steps, samples = (200, 129) if full20 else (30, 33)
    # Warm the optimizer's lazy imports before recording per-chart allocations.
    warm = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.Adam([warm])
    warm.grad = torch.autograd.grad(warm.square(), warm)[0]
    optimizer.step()
    rows = []
    for seed in seeds:
        charts = ("ordinary", "confluent") if seed % 2 == 0 else ("confluent", "ordinary")
        for chart in charts:
            rows.append(train_chart(seed, chart, steps=steps, samples=samples))
    summary = {}
    for chart in ("ordinary", "confluent"):
        chosen = [row for row in rows if row["chart"] == chart]
        summary[chart] = {key: statistics.median(row[key] for row in chosen)
                          for key in ("final_mse", "final_linf", "seconds", "sigmoid_calls", "python_peak_bytes",
                                      "parameter_gradient_optimizer_bytes")}
        summary[chart]["seeds_attaining_boundary"] = sum(row["first_attained_boundary_step"] is not None for row in chosen)
    return {"schema_version": 1, "mode": "full20" if full20 else "smoke", "seeds": list(seeds),
            "dtype": "float64", "device": "cpu", "optimizer": "Adam", "learning_rate": 0.01,
            "budget": {"steps_per_chart_seed": steps, "samples": samples, "parameters_per_chart": 4},
            "scope": "finite numerical experiment; near-series truncated at 6 terms with explicit remainder API",
            "reference_scope": "80-digit Decimal finite centered formula, rounded once to float64; numerical reference, not an interval proof",
            "memory_scope": "Python peak excludes backend tensor allocator; explicit parameter/gradient/optimizer bytes exclude graph temporaries",
            "timing_scope": "wall time includes identical instrumentation; step budgets equal, elapsed time is measured",
            "conditioning": conditioning_scan(), "boundary_demo": attained_boundary_demo(),
            "summary": summary, "trials": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full20", action="store_true", help="twenty fixed seeds with 200 equal optimizer steps per chart")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts")) / "neuromanifold_collisions" / ("full20.json" if args.full20 else "smoke.json")
    report = run(full20=args.full20)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "summary": report["summary"], "boundary_demo": report["boundary_demo"]}, indent=2))


if __name__ == "__main__":
    main()
