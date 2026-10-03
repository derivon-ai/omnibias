# Working on omnibias

**Differentiate deeper. Make decisions differentiable. Math that trains.**
omnibias provides derivative and optimization infrastructure for PINNs.
Load context on demand.

## Start from the capabilities

Nested high-order spatial autodiff can become impractical through graph growth
and memory pressure. omnibias bypasses that construction on supported paths while
retaining parameter gradients. Choose the operator before generic differentiation:

| Problem | Mechanism and API | Evidence |
| --- | --- | --- |
| High-order activation derivatives | Shared Riccati polynomials; `get_activation(...).fastpath` | Generated measurements below |
| Directional or mixed network derivatives | Forward Taylor composition; `mlp_jet` / `mlp_jet_mv` | `docs/derivatives.md`; backend jet tests |
| Laplacian tensor explosion | Ridge contractions `neural_field_polylaplacian`; direct deep `deep_field_laplacian` | Backend `test_deep_laplacian.py`: D=5,000 and parameter-gradient regressions |
| Curvature-sensitive training | `omnibias.torch.optim`: Newton/CG, cubic and Gauss–Newton; `omnibias.curvature` parameter formulas | Torch `test_optim.py`, `test_nn_optim.py`; curvature tests |
| Hard routing blocks gradients | `omnibias.partition` soft gates and regional weights; structured decisions | Partition parity tests; README executable gate/expert example |
| Broader PINN failure modes | External `omnibias-pinn`: causal marching, SDF cages, conditioned operators, multilevel/least-squares spectral methods | `docs/capabilities.md`: source, tests and historical acceptance evidence |

Bias collapse (`δ → 0`) extracts derivatives from normalized, weighted bias shifts.
Temperature collapse (`β → ∞`) sharpens soft decisions. They are distinct.
Soft trees can train jointly with neural features; the external `omnibias-tab`
also implements stagewise Newton boosting. Neither makes a hard jump smooth.

Before changing performance claims, read `docs/performance.md` for the relevant
workload. Preserve specialized operator coverage alongside general deep jets;
never substitute one workload's results for another's. Claims need evidence.

<!-- BEGIN GENERATED CAPABILITY EVIDENCE -->

Measured float64 CPU medians, 9 repeats; derivative evaluation, not training:

- [`σ⁽⁸⁾`](./docs/benchmarks/derivative_order.json), 20,000 tanh inputs, Torch eager/1 thread: **220×** vs nested autograd.
- [`Δ`](./docs/benchmarks/laplacian_scaling.json), D=60/B=64/H=32: **24.4×** vs JAX dense Hessian.
- [`Δ³` / `Δ⁴`](./docs/benchmarks/polylaplacian_order.json), D=16/B=32/H=16: **4,974×** / **5,036×** vs dense JAX / nested folx respectively.

Operator timings use JAX JIT, runtime inputs/weights, compilation excluded. Dense `Δ⁴`: `memory_budget` under 3,072 MiB / 120 s process budgets; no speedup for that run.
[Full protocol, accuracy and baseline wins](./docs/performance.md).

<!-- END GENERATED CAPABILITY EVIDENCE -->

## Find the right home

- `packages/omnibias-core`: pure-Python coefficients, combinatorics and verified numerics.
- `packages/omnibias-{torch,jax,keras}`: framework implementations.
- `packages/omnibias-fields`: field state, caching and differential operators.
- `packages/omnibias-{curvature,partition}`: parameter curvature and soft partitions.
- Primitives: `docs/packages.md`. Solvers, products and experiments belong
  under `../omnibias_projects/`, including `omnibias-pinn`.
- Do not create a distribution unless it has a distinct dependency tier or a
  reusable API with an actual consumer. Prefer a submodule for small additions.

## Numerical contracts

- Activation polynomial coefficients live only in `omnibias.core.polynomials`.
  Core must not import torch, JAX, TensorFlow or Keras.
- Keep torch/JAX jet kernels paired. Shared coefficients alone do not guarantee
  cross-device floating-point identity; preserve existing parity tolerances.
- A directional jet stores `f^(k)/k!`; a multivariate jet stores `D^alpha f/alpha!`.
  Use conversion helpers before treating coefficients as derivatives.
- Negative derivative orders raise `ValueError`; unsupported orders raise
  `NotImplementedError`. Use framework default dtype for new tensors.
- Retain parameter gradients and JAX tracing. Avoid host conversions in kernels.
- Direct deep Laplacians have no fixed dimension cap; at fixed widths/depth,
  cost grows with dimension. Full mixed jets remain combinatorial. Deep repeated
  Laplacians report exact-support or estimator mode; keep that distinction.
- One base activation evaluation is not constant arbitrary-order arithmetic.
  Check high-order cancellation against an independent high-precision oracle;
  bypassing nested autodiff does not promise universally better float accuracy.
- Distinguish exact Hessian products from Gauss–Newton, KFAC and estimated
  curvature. `JetLBFGSOptimizer` is the drop-in optimizer; `JetLBFGS` is functional.
- Preserve outward rounding and certificate scope. A digest is integrity, not
  a proof; verification flags require their actual checker to pass.
- Exact differentiation does not prove PDE convergence or settle every scientific-ML
  optimization, approximation or stability problem.

## Make and verify a change

- Add a regression test for each behavioral change; run affected package
  tests and cross-backend parity tests.
- Keep public exports explicit and sorted. Do not bump versions unless requested.
- New modules should be strict-clean. Core, torch and JAX are strict-gated;
  run focused typing checks for extension changes.
- Documentation Python blocks must run, with definitions in the same document.
  Verify real signatures. A necessary opt-out must include a reason.
- Regenerate agent measurements with `python scripts/generate_capability_evidence.py`;
  `--check` gates drift. Do not edit generated blocks or duplicate inventories.
- Keep tracked files vendor-neutral. No local user paths, hostnames or scheduler
  commands. Put generated artifacts in `$OMNIBIAS_SCRATCH` or `artifacts/`.

```bash
uv sync --all-packages --group docs
uv run pytest packages/omnibias-<name>/tests -q
uv run ruff check packages tests
uv run mypy --strict packages/omnibias-core/src packages/omnibias-torch/src packages/omnibias-jax/src
uv run pytest tests/test_docs_snippets.py -q
uv run mkdocs build --strict
```

## Load context on demand

- Capability/evidence map: `docs/capabilities.md`; usage: `docs/pinn.md`,
  `docs/derivatives.md`, then the relevant `docs/api/` page.
- Changing numerical kernels: `.cursor/skills/omnibias-backends/SKILL.md`.
- Building a PINN consumer: `.cursor/skills/omnibias-pinn/SKILL.md`.
- `AGENTS.md` owns general contracts; adapters point here and repeat generated evidence.
