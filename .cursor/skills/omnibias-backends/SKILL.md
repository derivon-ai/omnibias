---
name: omnibias-backends
description: Choose or extend omnibias derivative kernels, including high-order jets and direct Laplacians, while retaining parameter gradients and backend contracts.
---

Read `AGENTS.md` for contracts and `docs/derivatives.md` for signatures.
The key advantage is bypassing nested spatial-autodiff graph/memory growth
while keeping the result differentiable in model parameters.

| Requested quantity | Choose first |
| --- | --- |
| Activation derivative | `get_activation(...).fastpath` |
| Selected direction / mixed partials | `mlp_jet` / `mlp_jet_mv` |
| One-layer Laplacian or iterate | `neural_field_laplacian` / `neural_field_polylaplacian` |
| Deep Laplacian without a dense Hessian | `deep_field_laplacian`; D=5,000 regression |
| Deep higher iterate | `deep_field_polylaplacian_with_report`; inspect exact/estimated mode |

Use shared core coefficients and paired tensor kernels; preserve normalization,
tracing and parameter gradients. Validate against an independent analytic or
high-precision oracle, then test parameter gradients and backend tolerances.
Read `docs/performance.md` before benchmarking or editing claims; distinguish
activation, ridge-operator and general deep-jet workloads. Report compilation,
steady-state evaluation and training separately. High-order floating-point
cancellation remains possible even when nested autodiff is eliminated.

<!-- BEGIN GENERATED CAPABILITY EVIDENCE -->

Measured float64 CPU medians, 9 repeats; derivative evaluation, not training:

- [`σ⁽⁸⁾`](../../../docs/benchmarks/derivative_order.json), 20,000 tanh inputs, Torch eager/1 thread: **220×** vs nested autograd.
- [`Δ`](../../../docs/benchmarks/laplacian_scaling.json), D=60/B=64/H=32: **24.4×** vs JAX dense Hessian.
- [`Δ³` / `Δ⁴`](../../../docs/benchmarks/polylaplacian_order.json), D=16/B=32/H=16: **4,974×** / **5,036×** vs dense JAX / nested folx respectively.

Operator timings use JAX JIT, runtime inputs/weights, compilation excluded. Dense `Δ⁴`: `memory_budget` under 3,072 MiB / 120 s process budgets; no speedup for that run.
[Full protocol, accuracy and baseline wins](../../../docs/performance.md).

<!-- END GENERATED CAPABILITY EVIDENCE -->
