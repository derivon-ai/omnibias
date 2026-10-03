# PINN derivative benchmarks

Run from the repository root with the PyTorch/JAX development dependencies
installed. The two Laplacian comparison scripts also require `folx`.

| Command | What it measures |
| --- | --- |
| `uv run python benchmarks/readme_derivatives.py` | Bounded deep-network derivatives: nested AD, omnibias Riccati jets and JAX Taylor-mode AD, with separate compile times and an independent 80-digit reference. |
| `uv run python benchmarks/derivative_order.py` | Activation derivatives versus nested autodiff and finite differences, orders 1–8. |
| `uv run --with folx python benchmarks/laplacian_scaling.py` | One-layer Laplacian cost versus dimension; compares Hessian traces and `folx`. |
| `uv run --with folx python benchmarks/polylaplacian_order.py` | Repeated Laplacian cost and accuracy versus derivative order. |
| `uv run python benchmarks/jet_vs_nested_ad.py` | Directional-jet residuals and parameter optimization on a manufactured Poisson problem. |
| `uv run python benchmarks/deep_laplacian_scaling.py` | Deep Laplacian accuracy, dimension limits, cost and backend parity. |

The last two commands accept `--full`; their default runs write `*_smoke.json`.
The first three use their fixed script configurations and have no CLI flags.
Every JSON output goes to `$OMNIBIAS_SCRATCH`, defaulting to repository-relative
`artifacts/`. Full runs use distinct filenames in the same output directory.

Compare results only at matching dtype, accuracy, model size, derivative order
and timing protocol. Scripts record configuration and environment provenance;
a failed accuracy or cost threshold must remain a failure in any report.
Higher-order baselines can be expensive. A timing win does not establish PINN
training convergence or continuous-domain error bounds.

The deep-MLP measurement is retained in `docs/benchmarks/readme_derivatives.json`.
It includes slower results as well as faster ones, and times derivative
evaluation without parameter backward. PyTorch uses eager execution and one
CPU thread; JAX uses JIT and its runtime's default CPU thread policy. Compare
within a backend, and include compilation for workloads that cannot amortize it.

To refresh the public figure after reviewing a new measurement, copy the small
JSON from the scratch directory to that documented location, then render it:

```bash
uv run --no-project --with matplotlib python benchmarks/render_readme_benchmark.py \
  docs/benchmarks/readme_derivatives.json docs/img/derivative-benchmark.svg
```

Update numerical documentation claims together with the artifact and figure. The
activation-order experiment also uses independent `mpmath.diff` references;
its accuracy metrics are sampled at 33 positions with 80 decimal digits,
while timing and backend comparisons cover the full batch.

The flagship specialized comparison uses three separate workloads; the deep
MLP jet experiment does not substitute for any of them. The two spatial
scripts pass coordinates and all parameters as **runtime JIT arguments**.
The earlier zero-argument closures allowed XLA to precompute their answers.
`laplacian_scaling.json` includes an optimized-HLO audit demonstrating that
constant folding on the recorded compiler. Compilation and first execution
are separate from the warmed steady-state samples in the new measurements.

`polylaplacian_order.py` runs each method/order in a fresh subprocess, with a
120-second wall-clock limit and a 3 GiB observed-RSS limit enforced by its
parent. It requires Linux `/proc` for memory monitoring. A budget termination
is reported explicitly and receives no estimated timing or speedup. All
completed methods must pass a three-point, 80-digit `mpmath.diff` reference;
activation derivatives use 33 reference points. Both use float64 inputs and
record exact source hashes, software versions, workload sizes and raw samples.

After rerunning all three scripts, copy their reviewed JSON artifacts to
`docs/benchmarks/` and regenerate the specialized figure:

```bash
uv run --no-project --with matplotlib python benchmarks/render_specialized_benchmarks.py \
  docs/benchmarks docs/img/specialized-derivatives.svg
```

The [performance guide](../docs/performance.md) explains workload differences,
execution policies, compilation costs and the scope of the published claims.
