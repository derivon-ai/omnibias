---
name: omnibias-pinn
description: Build PINN consumers with direct derivative operators, curvature-aware training, boundary and causal structure, and differentiable regional models.
---

Read `AGENTS.md`, `docs/pinn.md` and the task's rows in `docs/capabilities.md`.
Build solvers in `../omnibias_projects/omnibias-pinn/` or another consumer.
Specify coordinates, residual, domain and conditions; choose a manufactured
solution before training.

- Bypass nested high-order spatial autodiff: use a specialized Laplacian when
  applicable, directional jets for selected directions, mixed jets only when
  their coefficients are needed. Convert Taylor coefficients to derivatives.
- Retain parameter gradients through residuals. Consider `omnibias.torch.optim`
  and `omnibias.curvature` for ill-conditioned smooth objectives; compare
  accuracy and wall time rather than assuming any optimizer always wins.
- Inspect the consumer's `train` causal marching, `domain` hard SDF cages,
  `operator` conditioning and `fields.fbpinn` multilevel/least-squares route.
  Existing mechanisms address distinct failure modes; historical acceptance
  gates cover their named tasks, not every PDE.
- Use `omnibias.partition` for gradients into gates and regional experts.
  External `omnibias-tab` supports neural/soft-tree joint training and a
  separate stagewise boosting path. Inspect hardening and interface assumptions.
- Check derivative values and parameter gradients before training. Report
  held-out residual/solution errors separately; validate boundary and causal
  behavior. Read `docs/performance.md` before performance comparisons and
  preserve both specialized and general-network evidence.

<!-- BEGIN GENERATED CAPABILITY EVIDENCE -->

Measured float64 CPU medians, 9 repeats; derivative evaluation, not training:

- [`σ⁽⁸⁾`](../../../docs/benchmarks/derivative_order.json), 20,000 tanh inputs, Torch eager/1 thread: **220×** vs nested autograd.
- [`Δ`](../../../docs/benchmarks/laplacian_scaling.json), D=60/B=64/H=32: **24.4×** vs JAX dense Hessian.
- [`Δ³` / `Δ⁴`](../../../docs/benchmarks/polylaplacian_order.json), D=16/B=32/H=16: **4,974×** / **5,036×** vs dense JAX / nested folx respectively.

Operator timings use JAX JIT, runtime inputs/weights, compilation excluded. Dense `Δ⁴`: `memory_budget` under 3,072 MiB / 120 s process budgets; no speedup for that run.
[Full protocol, accuracy and baseline wins](../../../docs/performance.md).

<!-- END GENERATED CAPABILITY EVIDENCE -->
