# PINN derivative benchmarks

Run from the repository root with the PyTorch/JAX development dependencies
installed. The two Laplacian comparison scripts also require `folx`.

| Command | What it measures |
| --- | --- |
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
