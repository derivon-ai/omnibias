# PINN derivative benchmarks

Run from the repository root with the PyTorch/JAX development dependencies
installed. The two Laplacian comparison scripts also require `folx`.

| Command | What it measures |
| --- | --- |
| `uv run python benchmarks/readme_derivatives.py` | Bounded deep-network derivatives: nested AD, omnibias Riccati jets and JAX Taylor-mode AD, with separate compile times and an independent 80-digit reference. |
| `uv run python benchmarks/derivative_order.py` | Activation derivatives versus nested autodiff and finite differences, orders 1–8. |
| `uv run python benchmarks/laplacian_scaling.py` | One-layer Laplacian cost versus dimension; compares Hessian traces and `folx`. |
| `uv run python benchmarks/polylaplacian_order.py` | Repeated Laplacian cost and accuracy versus derivative order. |
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

The README measurement is retained in `docs/benchmarks/readme_derivatives.json`.
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

Update numerical README claims together with the artifact and figure. The
activation-order experiment also uses independent `mpmath.diff` references;
its accuracy metrics are sampled at 33 positions with 80 decimal digits,
while timing and backend comparisons cover the full batch.
