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
| `uv run python benchmarks/operator_compiler.py` | Automatic ridge, disjoint-branch and affine-bottleneck lowering, with accuracy-gated Taylor/operator baselines and derivative-only training. |

`jet_vs_nested_ad.py` and `deep_laplacian_scaling.py` accept `--full`;
their default runs write `*_smoke.json`. The operator comparison has its own
family/order/method controls and does not use `--full`.
Every JSON output goes to `$OMNIBIAS_SCRATCH`, defaulting to repository-relative
`artifacts/`. Full runs use distinct filenames in the same output directory.

Compare results only at matching dtype, accuracy, model size, derivative order
and timing protocol. Scripts record configuration and environment provenance;
a failed accuracy or cost threshold must remain a failure in any report.
Higher-order baselines can be expensive. A timing win does not establish PINN
training convergence or continuous-domain error bounds.

The experimental operator comparison defaults to 20 synchronized samples in
each of three fresh processes. It stores exact implementation snapshots,
runtime operands, compilation/first-call costs, warm forward and training
samples, process RSS, and a separate untimed CPU tensor-memory profile.
Protocol v2 precomputes independently verified references outside the engine
worker budget, keyed by exact operands and request, and records both their cost
and total elapsed time. `--cache cold` starts with empty isolated compiler caches
and permits normal reuse within each worker; `--cache disabled` is the separate
cache-disabling diagnostic. The default allows persistent caches.
Optional `torch-jet` and `folx` baselines are used only when already installed.
The [operator guide](../docs/operator-compiler.md) describes the four workload
families, applicability limits, and reproduction commands. Failed accuracy,
timeouts, insufficient sample counts, or changed sources disqualify speedup
comparisons. The fastest successful applicable baseline is always reported.
Use `operator_report.py` to verify snapshot hashes and bundle raw reports,
source files, license notices and diagnostic attachments. The
[v2 measured results](../docs/operator-compiler-v2-results.md) keep different source
snapshots separate and include baseline wins.

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

## Clean environment and CI smoke

The locked `benchmarks` dependency group supplies folx, mpmath, NumPy and plotting
requirements. From a clean checkout, use Python 3.12 and CPU execution:

```bash
uv sync --locked --all-packages --group benchmarks --python 3.12
JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 uv run --no-sync python benchmarks/smoke.py
```

The smoke command reuses the activation, deep-jet and Laplacian comparison
implementations at bounded sizes. It covers Torch nested AD, JAX nested AD,
JAX Taylor mode and folx, and retains the independent high-precision checks.
Its outputs go to `artifacts/benchmark-smoke/`; they never update reviewed charts.
A full comparison is explicitly invoked with the commands above. Hardware,
compiler and dependency changes can alter both timing and rankings. Reproducing
an experiment does not promise an identical speedup ratio on another machine.
