# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded activation-only CUDA fusion diagnostic; writes JSON to stdout.

Run with an installed CUDA-enabled PyTorch and omnibias-torch. Compilation and
first execution are separate from warmed CUDA-event medians. This is not a
custom-kernel comparison or an end-to-end training benchmark.
"""
from __future__ import annotations

import json
import statistics
import time

import mpmath as mp
import torch
from omnibias.torch.activations import get_activation


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this diagnostic")
    mp.mp.dps = 80
    spec = get_activation("tanh")
    x = torch.linspace(-2, 2, 20_000, device="cuda", dtype=torch.float64)
    rows = []
    for order in (4, 8):
        def evaluate(value: torch.Tensor, n: int = order) -> torch.Tensor:
            return spec.fastpath(value, n)

        compiled = torch.compile(evaluate, fullgraph=True)
        start = time.perf_counter()
        compiled(x)
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - start
        timings = {}
        errors = {}
        gradient_errors = {}
        sample = torch.tensor([-.9, -.3, 0., .3, .9], device="cuda", dtype=x.dtype)
        oracle = torch.tensor([float(mp.diff(mp.tanh, float(z), order)) for z in sample],
                              device="cuda", dtype=x.dtype)
        derivative = torch.tensor([float(mp.diff(mp.tanh, float(z), order + 1)) for z in sample],
                                  device="cuda", dtype=x.dtype)
        for name, function in (("eager", evaluate), ("compiled", compiled)):
            errors[name] = float((function(sample) - oracle).abs().max())
            point = sample.clone().requires_grad_(True)
            grad = torch.autograd.grad(function(point).sum(), point)[0]
            gradient_errors[name] = float((grad - derivative).abs().max())
            torch.testing.assert_close(function(sample), oracle, rtol=1e-9, atol=1e-8)
            torch.testing.assert_close(grad, derivative, rtol=1e-9, atol=1e-7)
            for _ in range(5):
                function(x)
            torch.cuda.synchronize()
            samples = []
            for _ in range(25):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                function(x)
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end))
            timings[name] = statistics.median(samples)
        rows.append({"order": order, "median_gpu_event_ms": timings,
                     "compile_and_first_call_seconds": compile_seconds,
                     "max_absolute_error_80_digit_oracle": errors,
                     "max_gradient_absolute_error_80_digit_oracle": gradient_errors})
    print(json.dumps({"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                      "dtype": "float64", "points": 20_000, "input_interval": [-2, 2],
                      "warmups": 5, "repeats": 25, "oracle_points": [-.9, -.3, 0, .3, .9],
                      "scope": "activation-only forward runtime; sample gradient correctness",
                      "results": rows}, indent=2))


if __name__ == "__main__":
    main()
