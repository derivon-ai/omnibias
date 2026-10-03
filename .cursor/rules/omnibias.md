---
description: omnibias repository guidance
alwaysApply: true
---

Read `AGENTS.md`, especially “Start from the capabilities,” before choosing
derivative, PINN or decision architectures. It owns the numerical contracts.
Read `docs/performance.md` before editing performance claims: specialized
activation/Laplacian paths and general deep jets are different workloads.

<!-- BEGIN GENERATED CAPABILITY EVIDENCE -->

Measured float64 CPU medians, 9 repeats; derivative evaluation, not training:

- [`σ⁽⁸⁾`](../../docs/benchmarks/derivative_order.json), 20,000 tanh inputs, Torch eager/1 thread: **220×** vs nested autograd.
- [`Δ`](../../docs/benchmarks/laplacian_scaling.json), D=60/B=64/H=32: **24.4×** vs JAX dense Hessian.
- [`Δ³` / `Δ⁴`](../../docs/benchmarks/polylaplacian_order.json), D=16/B=32/H=16: **4,974×** / **5,036×** vs dense JAX / nested folx respectively.

Operator timings use JAX JIT, runtime inputs/weights, compilation excluded. Dense `Δ⁴`: `memory_budget` under 3,072 MiB / 120 s process budgets; no speedup for that run.
[Full protocol, accuracy and baseline wins](../../docs/performance.md).

<!-- END GENERATED CAPABILITY EVIDENCE -->
