# Changelog

## 0.5.0 / Fields 0.2.0 (2026-11-17)

- Promoted core, JAX and Torch to stable `0.5.0`, and Fields to stable `0.2.0`.
  Restored `omnibias.core.ladder`, `omnibias.core.local_jet`, the JAX and Torch
  local-training adapters, and `omnibias.torch.information` so published FermiNet,
  PINN and Geometry feature probes retain their supported imports.

- Focused this repository on 16 reusable primitive distributions for PINN
  derivative computation and supporting numerical operations.
- Extracted consumer distributions into standalone repositories under the
  sibling `omnibias_projects` directory, preserving their source and tests.
- Replaced the accumulated documentation and agent catalogs with executable
  derivative/PINN guides, a package map and two focused skills.
- Preserved the previous documentation and development records in the external
  `omnibias-research` repository.
- The JAX and PyTorch deep-field Laplacians now refuse `prelu` without
  `allow_almost_everywhere=True`, as they already did for `leaky_relu`: its
  second derivative is zero almost everywhere.
- Added the opt-in `omnibias.jax.experimental` subpackage. It provides
  closed-form one-layer ridge operators and their tensor-grid losses,
  Gauss–Newton steps and separable-network tables. For deep MLPs it adds
  first-layer projections of Laplacians and bi-Laplacians, divergences, and
  Sylvester log-determinants. The projected operators take `coordinates`, a
  diagonal diffusion and drift directions for time-dependent fields; the
  Laplacian also takes a state-dependent diagonal diffusion, one per point.
  The bi-Laplacian also returns the squared gradient norm. It also forms the Gram
  matrix of PDE residual gradients over all deep-MLP parameters, and the damped
  Gauss–Newton step from it, without building the Jacobian. Each name has measured,
  scoped evidence in `docs/performance.md`, including where it does not help.
- Extended the experimental residual Gram. `projected_gram` forms the Gram matrix
  and Gauss–Newton step for inputs wider than the network (`D > H`) without the
  first-layer rows, and for fourth-order blocks (`order=4`: Kuramoto–Sivashinsky,
  Cahn–Hilliard, hyperviscous advection) through the bi-Laplacian streams.
  `deep_residual_ntk_diagonal` and `projected_residual_ntk_diagonal` return the
  exact diagonal of the residual kernel, `|∂r_b/∂θ|²`, in `O(N)` memory.
  `pad_block` and `masked_residual` pad a block of rows to a fixed shape with
  exactly zero rows and columns of `K` and of the step. Their measured gains,
  null and partial results are in `docs/performance.md`: 2.95× geometric mean
  over the strongest baseline at `D / H ≥ 16` (16 of 18 comparisons above 2×
  by lower95), 3.78× for fourth-order blocks at `ρ ≥ 3`, 2.96× for the diagonal
  (parity at 32 streams per row) and about 2.3× for the projected diagonal.
- Added the compile-time guidance to `docs/performance.md`: cold compilation of
  the deep modules is dominated by per-shape GEMM autotuning rather than by depth.
  Disabling Triton GEMMs or autotuning shortens compilation by about 3–4× (medians) at a measured
  steady-state cost, and a persistent cache removes it on a second process. No
  default was changed and the library sets no global XLA flag.
- Recorded two negative results: cyclic-offset stream pairing is exact but
  failed its held-out steady-state gate, so rows remain the default
  (`_PAIRING = "cyclic"` or `"auto"` select it), and a factorized
  physics-informed DeepONet trunk was no faster than the strongest generic
  program, so it was not added.
- Added two programs to `omnibias.jax.experimental` from a test of seven cost-theory
  predictions, each registered with its adoption gate before any measurement.
  `residual_solve` and `residual_solve_transpose` solve with the Jacobian of a
  residual block `x + MLP(x)` through the Woodbury identity at the narrowest layer:
  4.5–25.7× (lower95) over the strongest of dense, `jacfwd` and GMRES in the 18
  held-out rows with `D ≥ 4H`, and no gain at `D ≤ H`. `residual_inverse_newton`
  is exact; the contraction regimes in which it beats fixed-point iteration are measured below (`residual_inverse`).
  `projected_deep_trilaplacian` returns the tri-Laplacian and the terms of the PFC
  residual by composing the projected bi-Laplacian: 5.6–198× over the strongest
  full-input program in 20 of 20 held-out rows with `D ≥ 2H` (growth `D^2.6`–`D^3.1`),
  and 1.7–3.3× slower than a generic engine given the same projected columns.
- Added a canonical dense-chain descriptor and one residual demand interface.
  `omnibias.core.chain.ChainDescriptor` describes a bias-bearing MLP by widths, activations
  and dtype with a stable fingerprint, and `omnibias.core.demand.Demand` names the derivative
  quantities a residual reads (`value`, `grad`, `grad_sq`, `directional`, `lap`,
  `bilap`). In `omnibias.jax.experimental`, `Residual(fn, demand)` is lowered to
  whichever engine can serve it (the full-gradient or the projected one), which refuses a
  demand it cannot provide; `legacy_full_gradient` and `legacy_projected` wrap the old
  callbacks, which are still accepted unchanged. `ridge_layers`, `separable_axis_layers`,
  `layers_of` and `describe_layers` convert the three layer conventions of the package into
  one. The shared point, coordinate, activation-gate and slope handling of the deep-MLP
  programs now lives in one private module; the operations each program emits are
  unchanged (114 golden cases are bitwise equal, with identical jaxprs).
- Added the descriptor-driven planner and its first front end. `omnibias.core.planner` holds the
  operator request, an algorithm registry with applicability rules and work models, a per-device
  cost model, a versioned persistent plan cache, memory budgets, and a heuristic or measured
  (`find`) selection that records every refusal with its reason. In `omnibias.jax.experimental`,
  `compile_operator(model, example_args, operator=...)` reads a model's traced program, recognises a
  dense layer chain (confirmed numerically on the example), plans the Laplacian and its powers,
  divergence, `log|det(I + J)|` and `(I + J) y = rhs`, and returns a callable whose `explain()` states
  what matched, what was chosen, the predicted costs and what was refused. A model that is not a
  recognised chain runs on a generic Taylor-mode or nested-autodiff executor with the reason given.
  `match_dense_chain` exposes the matcher. `compile_residual(model, blocks, quantity=...)` does the same
  for the residual Gram, damped Gauss–Newton step and NTK diagonal: blocks are `(x, Residual)` pairs, the
  declared demand decides which of the projected streams, the full-gradient streams and explicit
  autodiff rows can serve them, the step comes back in the structure of the model's parameters, and a
  model that is not a recognised chain runs on autodiff rows. The dispatch-regret and cold-cost evidence
  is in `docs/operator-planner.md#dispatch-regret` (the gate failed, so the cost model carries generic defaults).
  Both return programs that compile in separate, timed `trace`, `lower` and `compile` stages
  (`warmup(...)` pays them ahead of the first call and `explain()["cold_cost"]` reports them). Opt-in levers,
  none on by default: `bucket="pow2" | "geometric" | int` on `compile_operator` rounds a batch up so
  that new batch sizes reuse an executable (padding rows are discarded; `pad_block` with `masked_residual`
  does the same for residual blocks), `compile_policy="fast" | "fastest" | "auto"` passes accelerator
  compiler options at compile (recorded, and ignored on a CPU backend), `enable_compilation_cache(path)`
  sets JAX's persistent cache for the process, and `op.export(...)` stores the traced program as
  StableHLO. Calls made inside `grad`, `vmap` or an outer `jit` trace into the caller's computation as before.
  `omnibias.torch.experimental.compile_operator` now serves both: an `OperatorSpec` still compiles the
  structural program, and a string operator (`"laplacian"`, ...) is planned as below.
- The JAX `deep_field_polylaplacian` in `auto` mode now uses the exact projected program for `k` in
  `{2, 3}` once the support count exceeds the budget, where it previously sampled; it samples only
  when that program refuses the activation or the layer shapes. `mode="structured"` selects the
  program explicitly. `PolylaplacianReport.mode` can therefore be `"structured"`. The performance audit
  below extends the same routing, from a measured input dimension, to the cases within the budget and to PyTorch.
- Made the experimental Gauss–Newton solves report and, on request, bound their conditioning.
  The dual step `J^T (K + d I)^-1 r` has error about `kappa ε` with `kappa <= (λmax + d) / d`, so a
  kernel that agrees across engines to 1e-12 can still give steps that differ at 1e-3 (the A31 audit).
  `compile_residual(..., conditioning="off" | "report" | "floor")` and
  `ConditioningPolicy` estimate `λmax` (power iteration, with a rigorous trace/Gershgorin upper bound),
  put `condition_estimate` and `expected_step_relative_error` in `explain()`, and with `"floor"` raise
  the damping to the smallest value that meets the requested step tolerance (`"report"`, the default,
  never changes a result; `"floor"` does, and says by how much). `projected_residual_gauss_newton` and
  `deep_residual_gauss_newton` take the same policy. `gram_precision="compensated"` forms the Gram
  matrix with an exact (Ozaki-style sliced) float64 product, correctly rounded for rows whose dynamic
  range fits the slice budget, and `compute_dtype="float64"` evaluates a float32 problem in float64
  (needs `jax_enable_x64`). Compiled programs now compare XLA's memory analysis with the plan's
  `memory_budget` before running and raise `MemoryBudgetExceeded`; `compile_operator` then retries the
  next ranked candidate and records the refusal and the retry in `explain()`. Property-based tests
  compare every feasible executor with the dense Hessian on random chains, and a slow variant with a
  50-digit oracle.
- Added the planner front ends to PyTorch, with the same registry, cost model, refusal reasons and
  `explain()` keys as JAX (`backend="torch"`). `compile_operator(model, example_args, operator="laplacian",
  ...)` (also `compile_planned_operator`) plans the Laplacian, `Delta^p`, divergence, `log|det(I + J)|` and
  `(I + J) y = rhs` for a layer list or an `nn.Sequential` of `Linear` and activation layers, using ports of
  the projected Laplacian, divergence, log-determinant and Woodbury programs (with `input_factor`'s
  Cholesky-or-Householder basis) next to forward Laplacian, nested `torch.func` and Hutchinson; gradients
  reach the weights and the inputs. A generic `nn.Module` runs on the reverse-over-forward or nested
  engine and says why. `compile_residual(model, blocks, quantity=...)` plans the residual Gram, damped
  Gauss–Newton step and NTK diagonal on the stream identity or on `torch.func.jacrev` rows
  (`projected_streams` is refused on PyTorch with that reason), with the conditioning policy now shared with JAX
  through `omnibias.core.conditioning`; `op.residual()` evaluates the residual alone and
  `op.parameter_diagonal()` the exact per-parameter Gauss–Newton diagonal. In `omnibias.torch.optim`,
  `PlannedGaussNewton` runs Levenberg–Marquardt on a planned program (three dampings from one Gram,
  acceptance by the cheap residual), `planned_gauss_newton_step` applies one step, and
  `DiagonalCurvature`, `ConformalSymplectic` and `FrugalCurvature` accept `gn_diagonal=planned_gn_diagonal(op)`
  in place of one backward pass per residual row. Projected `Delta^2` / `Delta^3` and the projected residual
  streams are not ported to PyTorch. `tests/test_cross_backend_fixtures.py` replays
  `omnibias/core/fixtures/cross_backend_operators_v1.json` (exact float64 weights, references computed in 90-digit
  `mpmath` by `scripts/generate_cross_backend_fixtures.py`; moved from `tests/data` so the conformance suite
  ships with the wheel) on both backends: they agree with the references to 1e-11 and with each other to 1e-12.
- A deep field with `_omnibias_dispatch == "jet_mlp"` can opt in to the planner by defining
  `planned_chain(state, names)` (a `PlannedChain` of its `(W, b, activation)` layers):
  `laplacian`, `biharmonic`, `polylaplacian` and `divergence` of `omnibias-fields` then run through
  `compile_operator` on either backend, with gradients into the layers. Fields without the hook keep their own
  methods, and `planned_routing(False)` switches the routing off in a scope. Nothing is cached on the
  `FieldState`, so the readout-independence contract is unchanged.
- Extended the planner to models that are not dense chains. The JAX matcher folds literal scales
  (`sin(w0 * (W x + b))` in SIREN) into the chain. A model that reads its input only through weight
  products `W x` (residual blocks, Fourier features, gated networks, attention projections) is served by
  the *subspace programs* on both backends, `subspace_jet` and `subspace_nested`: they differentiate along
  `min(rows, D)` directions, the columns of a thin QR basis of the weight-product row space, instead of
  every input coordinate, and are refused with the reason when the input is also read elsewhere or the
  rows are not fewer than `D`. The basis is held constant for differentiation, so first-order parameter
  gradients of the operator are exact and second-order ones are not. Measured against the generic engines
  (`docs/operator-planner.md`, `docs/benchmarks/planner/`): 2.5 to 54 times faster at `D = 4,096`, and
  1.1 to 2.4 times slower at `D` of 64 to 1,024 on an A100 and at `D = 64` on a CPU, where the work is
  launch-bound. An earlier singular-value basis was slower than the generic engines on an A100 and was
  replaced.
- `force=` now means what it says: a forced program that cannot run is an error with its reason, in
  `compile_operator` and `compile_residual` on both backends, instead of silently running another one (the
  silent retry had mislabelled forced baseline measurements). An unforced plan still retries the next
  feasible program and marks the plan mode `+retry` or `+memory_retry`.
- Added the generated operator coverage matrix (`docs/operator-coverage.md`, `docs/coverage/matrix_{jax,torch}.json`,
  `scripts/generate_coverage_matrix.py`): operator by architecture by activation by vector-field and
  residual form, each cell one of structured, reduced, fallback, composed, refused (with the reason) or
  not provided, checked against a dense-autodiff oracle to 1e-8 with no mismatch on either backend.
  `tests/test_coverage_matrix.py` fails if a committed cell disagrees or the page is stale.
- Added `docs/operator-planner.md`, a guide to requests, plans, `explain()`, subspace programs, residuals
  and Gauss–Newton, with the measured cost of the subspace programs including where they lose.
- Added a precision policy and its error tables (`docs/precision.md`, `omnibias.core.precision.PrecisionPolicy`,
  `docs/benchmarks/precision/`). Matrix products default to full precision (`highest`); TF32 is opt-in; a half-precision
  chain is refused for derivative order 2 and above unless `allow_low_precision=True`, and `compute_dtype="float64"` runs a
  float32 problem in float64. `explain()["precision"]` reports the policy and its notes. The tables give each program's error
  against a 50-digit oracle on small instances and against the same program in float64 on realistic ones, on JAX and PyTorch
  and on an A100 and a CPU; they describe benign random weights, not trained networks. On the JAX A100 runs TF32 was
  indistinguishable from full precision at tiny shapes and about three orders of magnitude worse at realistic ones, which is
  why the policy is explicit.
- Added data-parallel versions of the residual Gram, damped Gauss–Newton step and NTK diagonal over a device mesh
  (`distributed_residual_gram`, `distributed_residual_gauss_newton`, `distributed_residual_ntk_diagonal` in JAX; the same names
  in PyTorch over `torch.distributed`). Points are sharded and weights replicated; the devices all-gather compact stream factors
  rather than rows of the Jacobian. The damped solve runs on the replicated `K` and is not divided by the device count.
- `Plan.explain()` now reports `cost_model`: `"defaults"` for the generic cost model, or the description of the calibration that
  ranked the plan. Removed avoidable Python dispatch cost from the JAX compiled operator's call path (an array that already is a
  `jax.Array` is no longer converted on every call); results are unchanged.
- `compile_operator(..., operator="solve")` on JAX accepts example arguments with or without the right-hand side when the model
  itself takes only the input.
- Promoted the planned operators out of `experimental`: `omnibias.jax.operators` and `omnibias.torch.operators` are the stable
  front ends (`compile_operator`, `compile_residual`, the policies, `Plan` and `PlanCache`, the distributed functions and
  `conformance_engine`). The old `experimental` names are aliases that resolve to the same objects, silent until 0.6.0, a
  `DeprecationWarning` from 0.6.0 and removed in 0.7.0 (`omnibias.core.deprecation`); in PyTorch
  `experimental.compile_planned_operator` is now `operators.compile_operator`. Plans carry a format version
  (`PLAN_FORMAT`, `PLAN_MIGRATIONS`, `migrate_plan_config`): an older format is migrated, a newer one refused, and a cache file of
  another format is ignored. `omnibias.core.conformance` replays the public fixtures against any derivative engine and measures
  dispatch regret. See `docs/stability.md`.
- Interop. `register_engine(name, engine, operators=..., work=..., generic=True)` (both back ends, `unregister_engine` to remove
  one) registers any function-level derivative engine with the planner: it competes by predicted cost, is chosen by `force=name`, and
  appears in `explain()`; engines are generic, so they are refused for structured requests. `omnibias.jax.folx_compat` is a
  `folx.forward_laplacian` drop-in (same `FwdLaplArray` fields and `weights=`, agreeing with folx to roundoff; folx is not required
  for the drop-in itself). `omnibias.torch.deepxde_adapter` and `pina_adapter` evaluate DeepXDE and PINA Laplacians and divergences
  of their networks through the planner and match the library's own operators (a DeepXDE training run reproduces the library's loss history). Each adapter is tested against one
  release of its library, named in `docs/interop.md`, which also records where folx, DeepXDE or PINA are faster.
- Dispatch gate (A100-SXM4-80GB, float64): the cost model fitted on 152 calibration problems and frozen by hash was scored once on 110
  held-out problems against criteria registered before any measurement. **It failed the worst-regret criterion**: median regret 1.00,
  but one divergence problem is 64 times slower than the best program and six more are above 1.5 (the fallback, break-even and
  50-digit-oracle criteria hold). No calibration file ships, so every device runs the generic cost model, and `mode="find"` is the way
  to remove dispatch regret. The calibration and held-out streams were measured on different dispatch code, so a second round re-measured the
  calibration on the final tree, froze a refitted model and scored it once on a new set of 112 problems: **it failed the same criterion**
  (worst regret infinite because one first choice was never measured, and 23.9 without that cell; divergence again, `jacobian_trace`
  ranked first and 24 times slower than `projected`), with the other three criteria holding. A third round was not run. Both rounds are in
  `docs/operator-planner.md#dispatch-regret` and `omnibias_experiments/dispatch_gate_20261022/DEVIATIONS.md`.
- Added `docs/hardware.md`: device probes and the cross-validated per-device regret of the cost model on an A100 (40 and 80 GB), an
  H100, an A30 and a CPU. No device meets the rule for shipping a calibration; other GPUs, TPU and PyTorch are listed as unmeasured.
- Added `docs/distributed.md`: measured strong and weak scaling of the data-parallel residual Gram, Gauss–Newton step and NTK diagonal
  on 1, 2 and 4 A100s (JAX and PyTorch), including the replicated solve that bounds the scaling.
- Documented a limitation that a property test found: `jax.experimental.jet` raises `UnexpectedTracerError` on `jnp.logaddexp` (so on
  `jax.nn.softplus`) in JAX 0.10.0 and 0.10.2, which makes the `taylor_jet` program unusable for a function-model that calls it. The planner
  already refused that program at trace time and ran another; `force="taylor_jet"` raises. A reproducer is in `omnibias_upstream/jax-jet-logaddexp`
  (not filed). `docs/operator-planner.md` lists it under Limits and the property test states the one refusal it tolerates.
- Added an end-to-end training section to `docs/operator-planner.md` and the record `docs/benchmarks/planner/e2e_suite_a100.json`
  (checked by `tests/test_e2e_record.py`). On the pre-registered 13-problem JAX suite no problem-target row meets the registered headline
  (0 of 39; 6 losses; 13 rows where no pair reached the target; 11 rows reach the bound with compile time excluded), and a two-problem
  PyTorch subset (Adam, untuned) has none either, so the planner is documented as a per-call primitive and not as a demonstrated
  training speedup. Registration, traces and scoring: `omnibias_experiments/e2e_20261022/`.
- Prepared a reproduction kit outside this repository (`omnibias_experiments/repro_kit_20261022/`): nine audits recompute published tables and
  verdicts from raw records on a CPU, and five replications (conformance, precision tables, Laplacian against the dense trace, suite step
  cost, the XLA compile crash) re-measure on a GPU against bands fixed beforehand. A first internal replication (fresh session, one shared H100,
  one run) reproduced every audit and met R1, R3 and R5; **R2 and R4 were missed** (three precision records among 578, and both step-time ratios at
  1.00 with every median near 11 ms). The bands were not changed. On an exclusive A30 R2, R3 and R4 were met and the stall signature did not appear;
  repeats on an exclusive H100 were cancelled unrun after 17 minutes in the queue, so the H100 result stands as one shared-card run. The kit's verifier now prints every band and a caution
  for disturbed timings, and the kit records what changed after the replication in `REPLICATION_INTERNAL.md`. Nothing was published.
- Added the low-order programs and a training-aware planner (this round measured only the per-call grid; see the per-call section of
  `docs/performance.md` for what was and was not met). `omnibias.{jax,torch}.experimental.streams` is a stacked stream engine: the
  value, tangent and second-order streams of a deep MLP in one array, one `dot_general` and one elementwise update per layer, no
  factorization and no conditional, with directional jets to order four (`Delta` and `Delta^2`). `fused_streams` adds one-kernel
  forward and backward programs (Pallas on JAX, Triton on PyTorch behind `torch.library`, so `torch.compile` and CUDA graphs accept
  them); they are **forced-only** on JAX and in float64, where the measurements say they lose, and are refused where their shared-memory
  bound (166,912 B) is exceeded. The repaired `fused_closed_ridge` kernel uses the Pallas dot that exists in the installed version.
  The planner (`PLAN_FORMAT` 2; a format-1 configuration migrates) takes `differentiate` (`none`, `params`, `params2`), `accuracy`
  (with `exact=False`), `expected_uses="auto"`, `uses_hint`, `mode="auto"` and `safe_find`; ranks forward plus backward work and launches
  for `params`; admits SDGD, STDE and Hutchinson when an accuracy target allows them; and registers an installed `folx` as a candidate
  (`OMNIBIAS_FOLX_ENGINE=off` opts out). `compile_policy="auto"` retries XLA autotune and Triton-GEMM failures with autotune level 0 and
  then without Triton GEMM. `taylor_closed` serves `Delta` and `Delta^2`, **not** `Delta^3` as first planned: the streams carry order four
  at most and `Delta^3` needs order six, so the request is refused with that reason. `params2` ranks by three times the backward work and
  has no measurement behind it.
- Fixed a ranking defect found by the coverage run: the work model of `forward_polylaplacian` counted a handful of streams, but the
  program unrolls `sum_{s <= p} C(D, s)` local jets (one per support set), so at `D = 12` it ranked above `projected` for `Delta^2` and
  `Delta^3` and then spent minutes tracing and tens of gigabytes. It is now priced by its support jets and compile size, and refused
  above 2,000 supports with the reason; the coverage matrices select `projected` there again.
- Fixed a ranking defect of the residual-Gram work model, found by replaying the planner over the recorded dispatch-gate cells
  (`omnibias_experiments/dispatch_fix_20261101/`): the first-layer projection of `projected_streams` runs a QR (one `geqrf` and one
  `orgqr` library call, counted in the compiled CPU and A100 programs) whenever more than `H_1` inputs are selected, and the model charged it nothing, so
  the planner chose that program in all 56 distinct order-2 cells although `full_streams` was the best measured program in 45. The
  model now charges the QR's flops and two solver calls. Replay over the 382 distinct recorded cells (a ranking replay of older
  programs, in-sample, not a held-out gate): worst regret 4.25 to 2.71, 90th percentile 1.29 to 1.20, cells above 1.5 from 17 to 7,
  no pick fails. Seven cells stay above 1.5: three order-2 Gram cells (worst 1.66), three Laplacian cells and one log-determinant cell;
  the worst, 2.71, is a Laplacian cell that this change does not touch. A second
  correction, pricing the factorized first-layer block of the projected engine at three times the explicit rows of the full engine for
  `D <= H_1` (an operation count that matches the compiled program to 2 to 4 %), cut the worst order-2 Gram regret to 1.21 but raised
  two cells that were within 1.10 above it, so by the registered adoption rule it was **not adopted**.
- Added `compile_training_step` (JAX: sampler, planned operators, loss, gradient and optimizer update in one `lax.scan` executable with
  prefetched points and donated state; PyTorch: a CUDA-graph step), the second-order step modules `omnibias.{jax,torch}.optim_pinn`
  (row-budgeted Gauss-Newton with a dense or conjugate-gradient solve, Levenberg-Marquardt with a held-out trust region, ENGD's grid line
  search, SPRING momentum, KFAC-expand, L-BFGS with line-search counts, order-three and order-six Gram), the demand vocabulary
  (`QuantitySpec`, `streams_quantities`, `streams_residual`), `Skip` and `fourier_feature_layer` for chains, and the PyTorch ports of
  `projected_deep_bilaplacian` and `projected_deep_trilaplacian` (dense Hessian storage; the projected residual streams are still not
  ported). None of these is part of the stable surface.
- Robustness: `compile_policy="auto"` and `safe_find` above; a `TreeSpec` incompatibility with PyTorch 2.9; the Hutchinson probe key;
  the public surface of `omnibias.fields`; the `_experimental_stage` manifest. `docs/hardware.md` no longer names a scheduler.
  Known and not fixed: the `_experimental_a100` tree carries no SPDX headers and is untouched by design; the `handoff/` directory
  contains local paths and scheduler names that the leakage test reports; and `mypy --strict` could only be run here against
  PyTorch 2.9 stubs, not the locked 2.14.1, where it reports untyped-`Function` and `Any`-return findings in the PyTorch
  experimental modules that this tree's older modules share, so a run against the lock is still owed.
- Registered, not run: end-to-end suite v2 (`omnibias_experiments/e2e_v2_20261024/SUITE_V2.md`) and dispatch gate round three (Addendum C
  of `omnibias_experiments/dispatch_gate_20261022/PREREGISTRATION.md`; amendments A1 and A2 record two edits to registered files before
  any measurement). Snapshot tooling (`percall_20261024/snapshot.py`) builds the wheel snapshot those runs must measure and refuses the
  live tree. New environment switches: `OMNIBIAS_PALLAS_INTERPRET`, `TRITON_INTERPRET` (kernel tests on a CPU), `OMNIBIAS_FOLX_ENGINE`,
  `OMNIBIAS_PLAN_CACHE` and `OMNIBIAS_CACHE_DIR`.
- Measured the per-call grid on one A100 (`docs/performance.md#per-call-grid`, record `docs/benchmarks/percall_grid.json`,
  registration and 13 dated deviations D1 to D13 under `omnibias_experiments/percall_20261024/`). **No registered target was met as
  written.** T1 (planned default within 1.05 of the best exact baseline): no group of 24. T2 (2 times `folx` value-and-gradient, 1.5 times
  forward, in latency-bound cells): 0 of 60 evaluable groups; PyTorch is not evaluable because `folx` is JAX-only. T3 (1.5 times at orders three
  and four): 18 of 30 evaluable groups, met for order four against nested `folx` and collapsed Taylor mode in most cells and not for the order-three
  forward on JAX. T4 (ratios within 10 % of the published ones): 0.51, 0.10 and 0.10 of the published ratios on this host; the published
  revision and the working tree give the same omnibias times here, so the difference is the host's, not a regression. T5: 0 of 12 groups,
  and a CUDA-graph training step of 1.014 ms in float64 (missed) and 0.820 ms in float32 (met). Where the planned default wins: `D >= 2H`
  on JAX (median 3.0 times the best `folx` forward on the device, 27 times at `D = 4,096`) and `D = 100` to `1,000` training steps
  (2.9 times at `D = 1,000`); where it loses: the narrow cells (0.72 to 0.93 of `folx`) and every case that a launch floor decides.
  The Pallas kernel is slower than the XLA stream engine and than `folx` forward on JAX; the Triton kernel is 2.6 to 3.1 times faster than
  the planned PyTorch default at `D = 2` (float32) and the planner **does not choose it** (the ranking prices it third and `find`
  times two candidates; not tuned against this grid). A CUDA C++ kernel behind an XLA custom call was not built
  (`percall_20261024/KERNEL_DECISION.md`). Not run: a second card, suite v2, gate round three.
- Fixed from the grid: the planned bi-Laplacian under `jit` and parameter differentiation at `min(D, H) >= 32` (arrays made inside a trace were
  captured by a `custom_jvp` rule); `compile_training_step(graph="auto")` on PyTorch after a failed CUDA-graph capture (the default generator
  stayed flagged as capturing and the warm-up steps stayed applied); `taylor_closed` at `Delta^2` above 16,385 channels (it held more than 98 GiB of
  host memory) is refused with a pointer to `projected`; the `find` timers of both back ends record a candidate that raised (out of memory
  included) as failed instead of failing the plan; and the benchmark harness (double backward through `jet`, buffers held between methods, an
  eager polarization plan, a fixed 4 to 5.5 ms cost in short timing blocks). Open: an XLA segmentation fault compiling a 2.2 TB candidate under
  an unlimited `memory_budget` (use `safe_find=True` or a finite budget), one unexplained PyTorch crash in the TF32 candidate pool, stream
  engines at order three are not CUDA-graph capturable on PyTorch, and PyTorch's reserved memory after the capture pools is not explained.
- Added the full-scope evidence matrix (`docs/benchmarks/full_scope.json`, built by `scripts/build_full_scope.py`; summary in
  `docs/performance.md#full-scope-matrix`). Every cell of every benchmark record carries a verdict from its interval (big win,
  win, parity, loss, incomplete), its baseline kind and its source record, so wins, ties and losses against independent
  baselines are all visible. The capability evidence block of `.cursor/rules/omnibias.mdc` is generated from it.
- Shipped optimizations (`docs/performance.md#shipped-optimizations`). Each change was gated against an immutable snapshot of the earlier tree with paired,
  alternating fresh-process rounds: bit-identical where it claimed to be a refactor, inside the cell tolerance where a summation order
  changed, paired median at most 1.02 times the baseline, no more temporaries or CUDA peak. Public signatures and return types, `PLAN_FORMAT`
  and the `PlanCache` file format are unchanged, and every new option defaults to the old behavior.
  Faster, measured as `new / old` (float64, one CPU thread unless noted): polarization plans are sparse and memoized and contracted by gather
  and segment-sum (0.006–0.042 for six plan builds, outputs identical); PyTorch `deep_field_polylaplacian` in `auto` runs the exact projected
  program from `STRUCTURED_MIN_DIM` (0.02–0.30 forward, 0.01–0.23 gradient) and takes `mode="structured"`; the JAX one does the same from its own
  crossover where it used to enumerate supports (0.08–0.17 forward, 0.05–0.11 gradient, XLA temporaries 12.8 to 2.2 MB and 170 to 8.7 MB at the
  mid size), and the third power at mid size, which the earlier support enumeration could not finish in a 64 GB reservation, now takes about 10 ms
  forward in 22 MB; routed PyTorch fields (`laplacian`, `biharmonic`, `divergence`, one planner call for all vector components) 0.15–0.35 forward
  and 0.35–0.54 gradient; the PyTorch support enumeration runs one batched program per support size (0.10–0.34); JAX stream engines on wide
  inputs 0.46–0.57 forward; a JAX `second_order_streams` with a traced `hess_form` under `jit` carries `m` coordinate tangents with the metric applied
  in-graph (0.20–1.06 forward, 0.15–0.80 gradient on four CPU cells, XLA temporaries 0.15–0.67 of before; on an accelerator only for `m <= H/4`,
  where the Gram stack is faster above that); distributed Gram and Gauss–Newton programs are built once per signature instead of recompiled on every call
  (609 ms to 0.2 ms); `compile_training_step` 0.56–0.70. On one exclusive A100 the PyTorch support enumeration is 0.003–0.11 and the routed
  fields 0.24–0.55; on the same card JAX's deep polylaplacian cells take 0.27–0.29 forward and 0.04–0.12 gradient, the wide jets 0.85–0.97,
  the training step and run 0.84 and 0.76, and every other cell has the compiled program of the old tree. Parity elsewhere: the projected programs, Gram blocks and planned operators are 0.86–1.01 of the old time (the same compiled
  programs on JAX).
  Prices and decisions that stay visible: the PyTorch batched support enumeration raises the **forward-only** CUDA peak by 1.4 to 7.4 MiB in three
  cells while its gradient peak falls (an accepted trade, recorded in the protocol); JAX keeps the old unrolled support program up to
  `SUPPORT_UNROLL_MAX_COEFFS = 2,000` jet coefficients and batches above, because the batched program compiles 3 to 70 times faster but needs
  1.7 to 1.8 times the temporaries where the unrolled one compiles (`SUPPORT_UNROLL_MAX_COEFFS = 0` batches everywhere); the JAX Gram form's identity block (`identity_block=True`) and per-layer
  rematerialization (`remat=True`) are opt-in because they lower XLA temporaries and never lower time: on four CPU and four A100 Gram cells the
  identity block saves up to 6.7 times the value-and-gradient temporaries (4.75 to 0.71 MB, `D = 64`, three hidden layers, A100) and costs 1.05 to
  1.19 times the value-and-gradient time in the cells where its interval is above 1, `remat` saves up to 25 % of them (nine paired rounds each);
  CUDA-graph replay of the planned PyTorch operator (`graph=`) is opt-in because it needs static shapes.
  Behavior changes: the PyTorch fused kernel's `precision="default"` ran TF32 and `"tf32"` ran IEEE; `"default"` now follows
  `torch.set_float32_matmul_precision` (full precision unless the process opts out) and `"tf32"` is TF32, `"highest"` is unchanged (a caller that
  relied on `"default"` being fast asks for `"tf32"`); PyTorch structured results stay connected to the readout bias, so a gradient into a
  parameter the operator does not depend on is an exact zero, as under support enumeration, and `torch.autograd.grad` without `allow_unused`
  does not raise; the work model of `taylor_closed` for `Delta^2` counts `m^2` directions (first choice right in 12 of 28 cells instead of 8 on JAX
  and 6 instead of 2 on PyTorch; dated amendment A3 of the dispatch-gate registration, which also records the per-backend launch table that was
  considered and **not** applied); `find` runs the predicted leader first, cuts a follower slower than `FIND_CUT_FACTOR = 4` times the best,
  skips one whose predicted peak exceeds `host_memory_limit()`, journals finished candidates in `plans.json.journal` so a killed run resumes, times
  the callable the operator will run, and keys its cache on precision, `compute_dtype`, compile mode and a bucket of `expected_uses`.
  Also fixed: `importlib.metadata.version` at every compile, `lru_cache` wrappers that broke Dynamo tracing, a zero-copy numpy view of a JAX array
  that a cached operator kept, the generic Hessian block (one triangle, mirrored) and quadrature-rule conversion caches (keyed so a replaced
  parameter, a changed structure or a traced input misses), and the JAX training step, which splits its keys inside the block and samples only the batch `step()` uses (the random streams under a fixed seed are identical). Known and not fixed: the
  `_experimental_a100` SPDX headers, the leakage hits in `tests/test_percall_record.py` and `handoff/`, and the fields `test_public_surface` on a
  host where the packages are not installed.
- Stream layout and the training step (`docs/performance.md#stream-layout`). `DEFAULT_LAYOUT` of the JAX stream engines is now `"auto"`:
  the split layout (value, tangent and second-order channels in separate arrays; no concatenate or slice of the tangents) on a CPU and
  while a training program is traced, the stack for forward-only calls on an accelerator. `compile_training_step` traces its programs under
  the new `omnibias.jax.experimental.streams.training_layout()` context, and `resolve_layout` reports the layout a call will take; an explicit
  `layout="stack"` or `"split"`, or a `DEFAULT_LAYOUT` set to either, wins. Values agree to rounding; the compiled programs differ, so a
  program that fingerprints the HLO of a CPU call sees a new one. Measured on one CPU thread (nine paired fresh-process rounds, seven
  narrow cells): value-and-gradient 0.48 to 0.72 of the stack time in 14 of 14 cases, forward 0.63 to 0.88 in 11 of 14 and 1.09 in one
  (`D = 16`, four layers of width 128); against `folx` the narrow cells go from 26 losses in 28 axes to 6 (0.77 to 1.33), and the `D = 100`
  training step from 0.63 to 1.02 of `folx`'s speed in float64 and 0.63 to 1.05 in float32; on an A100 the `D = 100` training step goes from 0.77 to 1.05
  of `folx`'s speed in float64 and from 0.74 to 0.91 in float32 (still a loss), and `D = 1,000` is 3.09 and 2.24 times faster than `folx`. On an A100 the split layout took 0.50 to 0.88
  of the stack's value-and-gradient time in 25 of 34 cases and was slower in none, and 1.06 to 1.62 of its forward time in 7 cases, so
  forward-only calls on an accelerator keep the stack.
- Directional-jet layout (`docs/performance.md`, "Per-call grid"). `directional_jet_streams` takes `layout="stack"`, `"split"` or `"auto"`
  (`JET_LAYOUT`, default `"auto"`: the split layout on a CPU from `JET_SPLIT_MIN_POINTS = 128` points, the stack otherwise; an accelerator keeps
  the stack). The split layout keeps the value and each Taylor coefficient in its own array. Values agree to rounding; a CPU call at 128 points
  or more compiles a different program. Third-order directional derivative on one CPU thread: 0.59 to 0.91 of the stack's forward time and 0.53 to
  0.89 of its gradient time from 256 points; omnibias is ahead of nested `jax.jvp` and `jax.experimental.jet` in 24 of 24 comparisons with it.
  At 32 points the split layout's gradient takes 1.07 to 1.25 of the stack's, so the default keeps the stack there, and on an A100 the split
  layout's gradient takes 1.06 to 1.20 of the stack's.
- `omnibias.jax.folx_compat` and `compile_operator`: shorter first call (`docs/interop.md`, "folx drop-in first call"). The matched chain
  is confirmed against the model in one jitted program instead of one compile per primitive, the example the plan is built on is drawn on
  the host, and under `vmap` the plan of one point is built only when a function that cannot be planned as a batch needs it (a separate
  runner is keyed on the traced batch). Values and the steady-state call are unchanged (CPU, 9 paired fresh-process rounds of 101 calls:
  0.95 to 1.01 of the earlier steady-state time). Paired fresh-process rounds, `D = 16` to `4,096`: the first call is 1.7 to 3.0 times shorter on CPU (3.5 to 5.2 s down to
  1.2 to 3.1 s) and 1.5 to 2.4 times shorter on an A100 (6.1 to 9.5 s down to 2.5 to 6.2 s). It is still 1.8 to 8.5 times `folx`'s first
  call, and those cells stay in the matrix as losses.
- `residual_inverse(y, layers, method="auto")` inverts a residual block `x + MLP(x)` by fixed-point steps, by Newton steps with Woodbury solves
  (`residual_inverse_newton`), or by fixed-point steps until the observed contraction predicts more steps than a Newton phase costs
  (`switch_steps`, default `4 H_n` on a CPU and `H_n` on an accelerator; Newton restarts from the starting point when the fixed-point
  residual grew). Time to tolerance (`docs/performance.md#woodbury-solves-for-bottleneck-residual-maps`, 160 rows in
  `docs/benchmarks/farm_residual_inverse.json`): fixed point is 1.2 to 9 times faster than Newton where the contraction is weak, Newton is 1.7 to 14 times
  faster where it is strong and fixed point does not reach the tolerance in 400 steps in several cells, and `auto` stays within 1.5 of the
  better one in float64. Prices that stay visible: `auto` costs 1.05 to 1.5 times at weak contraction, is up to 1.5 times slower than
  `residual_inverse_newton` in 8 of 25 A100 float64 cells, and in float32 on an A100 is 2.9 to 7.8 times slower than fixed point in six
  cells (use `method="fixed_point"` there).
- Planner on a CPU (`docs/operator-planner.md#cpu-class-prior-and-counted-pytorch-launches`). `CostModel.for_device("cpu")` ranks JAX programs
  with constants set from one core's hardware (`CPU_CLASS_PRIOR`) instead of the A100-scale generic ones, so the flops decide a ranking
  that launch and solver terms used to decide. Recorded dispatch regret (measured time of the chosen program over the best of the
  cell; two recorded sets of 76 Laplacian cells, one thread, float64; not re-measured): `D > H` median 1.31 and 1.61 (forward,
  value-and-gradient) before, 1.00 and 1.00 after; all cells 90th percentile 1.52 and 1.92 before, 1.20 and 1.18 after on the first set.
  Trade: at `D = 2`, width 32, two layers the prior picks `nested_autodiff` (value-and-gradient regret 1.07 and 1.23 before, 2.97 and
  1.90 after at `B = 1,024`); `force=` and `mode="find"` choose the other program. The A100 model is unchanged and a test pins it.
- PyTorch planner on an accelerator counts the CUDA kernels of one call (`TORCH_LAUNCH_SCALE`) where it counted fused XLA programs, and
  charges the fused Triton kernel its cold compilation (`TORCH_COMPILE_SECONDS`) over `expected_uses`. The planned default takes the
  kernel in float32 inside the region it was measured in (width at most 64, `D` at most 16) at about ten thousand expected uses.
- `omnibias.torch.experimental.streams.STREAM_LAYOUT` (`"auto"`, `"stack"`, `"split"`; argument `layout`): `"auto"` takes the split layout in
  eager calls (no concatenate or slice kernels) and the stack while a CUDA graph is captured or `torch.compile` traces
  (`docs/performance.md#pytorch-stream-layout`). The split layout leaves a readout bias out of the Laplacian graph; the program now ties it
  with a zero term so its gradient exists (a test pins stack and split gradients to each other).
  An affine first layer starts the second-order stream at zero in the split layout (a test pins affine layers in every position, and the
  gradients of every parameter, to the stack and to the dense Hessian; the root test `test_operator_fastpaths` found the earlier seed).
- JAX stream layout follows the pass: `layout="auto"` takes the split layout when the weights or points are being differentiated as the
  call is traced (`jax.grad`, `jax.value_and_grad`, `jax.jvp`), and a planned operator built with `differentiate="params"` traces its
  programs under `training_layout()`. A function traced first and differentiated afterwards (`jax.grad(jax.jit(f))`) is not seen: pass
  `layout="split"` or build the operator for parameter gradients.
- JAX `layout="auto"` takes the split layout in a forward-only float64 call on an accelerator with at most two hidden layers
  (`AUTO_SPLIT_MAX_HIDDEN = 2`; `resolve_layout` takes `hidden` and `dtype`). Registered comparison on an A100 (40 cells, four fresh-process
  rounds, forward device time of the 28 two-hidden-layer cells): float64 13 faster and none slower; float32 with TF32 14 faster and none
  slower; float32 with the highest precision 11 faster and one 5 percent slower (`D = 4,096`, `H = 64`, 32 points), so float32 keeps the
  stack and `layout="split"` selects the other side. Four-hidden-layer float64 cells: 2 of 12 slower, so deeper networks keep the stack.
  Eager timing of the two-hidden-layer cells is undecided in most cells (float64: 3 faster, none slower).
- PyTorch `STREAM_LAYOUT="auto"` takes the split layout in a captured CUDA-graph call that gradients flow through (grad mode on and a weight,
  bias or the points require grad); a captured forward-only call and `torch.compile` keep the stack. Replayed value-and-gradient time of
  the planned `differentiate="params"` program, 16 cells and three rounds per precision, new over old time: 0.51 to 0.94 in 38 of 48 cells
  where the interval decides, slower in none; the replayed forward time and the eager endpoints are unchanged (the flagged eager cells were
  level in a six-round re-run).
- New and re-measured farm records in `docs/benchmarks/` (`farm_percall_jax_a100.json`, `farm_percall_jax_cpu.json`,
  `farm_percall_torch_a100.json`, `farm_percall_torch_cpu.json`, `farm_time_to_target_rerun.json`) with the per-call grid of all 40 cells in
  float64 and in float32 with the highest and with TF32 precision, every baseline of the grid, and the end-to-end suite problems whose programs
  changed after it; the matrix takes the newer record where two cover a cell and keeps every loss with its interval.
- Log-determinant ranking: with the corrected work model of the planner's log-determinant programs, the eight recorded CPU cells choose a
  different program in every cell. Dispatch regret on those cells (JAX, float64, one thread; recorded default, then the corrected
  ranking): forward median 1.00 and 1.29, 90th percentile 3.77 and 6.20; value-and-gradient median 1.05 and 1.02, 90th percentile 5.26
  and 2.18. The forward pass pays, the training pass gains; `force=` and `mode="find"` choose the program.
- Full-scope matrix (`scripts/build_full_scope.py`): a farm record may declare `"supersedes": [<older record>]` and then replaces the older
  record on the cells they share (two farm records on one cell without it are an error), and rows marked `role=superseded_*` are replaced by
  the same record's `role=fix_verification` row of the cell.
- Exhaustive measured find and a training preset for the planner (`docs/operator-planner.md#planning-for-training`).
  `select`, `compile_operator`, `compile_planned_operator` and both `compile_residual` take `top_k`: `2` (default, as before) or `None`, which
  times every feasible candidate in the cost model's order, so a program the model ranks low can no longer be missed; the 4x cut-off and the
  host-memory skip still apply. `planned_evaluator` of the JAX and PyTorch training steps now defaults to `mode="find"` and `top_k=None`
  (before: `mode="auto"`, the two leading candidates once `steps >= 50`); pass `mode="auto"` or `mode="heuristic"` for the old behavior.
  Cache keys: a `find` key now carries the library versions (JAX or PyTorch, `omnibias-core`, the backend package) and a non-default `top_k`, so
  a plan measured under another release is measured again instead of reused (before, it was reused); plans cached under the old keys are no
  longer found and are re-measured once. A plan with a failure that repeats is now cached: a compiled size over the budget (JAX) and, with
  `safe_find=True`, a candidate whose process crashed are recorded as a refusal of the plan with a `find:` reason (`FindRefusal`), where before
  any failed candidate kept the plan out of the cache and every call timed it again; a timeout or an error at run time is still an infinite
  time and still not cached. Defect fixed: an exception while timing a program in the JAX (and PyTorch) `compile_residual` reached the caller;
  it is now a failed measurement listed in `explain()["executor_failures"]` and ranked last, as in `compile_operator`. `PLAN_FORMAT` is unchanged
  and `heuristic` ranking is unchanged. `benchmarks/percall_training_step.py` pins `mode="auto"` for `omnibias_planned` so that it keeps the
  protocol of the committed record. The `planner.py` change is recorded in Amendment A4(b) of the dispatch-gate registration (the `find`
  path only; the heuristic ranking, the cost model and the round-three gate are unchanged, and the manifest `ROUND3.sha256` verifies the file).
- SPRING momentum for the structured Gauss–Newton step: `projected_residual_gauss_newton` takes `momentum` and `prev_delta` (the previous
  returned step) and returns `J^T (J J^T + damping I)^{-1} (r - mu J d_prev) + mu d_prev`, the dual form `optim_pinn.gauss_newton_solve` and
  `LMConfig(momentum=...)` already use for an explicit Jacobian. `J d_prev` is read off the stored cotangent streams (the transpose of the
  step assembly), so no Jacobian row and no extra backward pass is formed; it agrees with per-row nested autodiff to 1e-9 on five problem
  shapes (tests in `test_experimental_projected_gram.py`), scalar or vector damping, eager or `jit`. `momentum` without `prev_delta` is refused.
- Regime guide `docs/regimes.md`, generated by `scripts/build_regime_guide.py` from `docs/benchmarks/full_scope.json` (no package imports; `--check`
  in CI beside `build_full_scope.py --check`): every measured regime by family, operator, backend, device, dtype and endpoint, with the cells that
  lose and the independent baselines that are faster listed next to the wins, and a loss to an omnibias variant read as "which variant won".
  `tests/test_regime_guide.py` checks that the page equals its rendering, that an incomplete cell is never a win and that its links resolve.
- Advisor skill `.agents/skills/omnibias-advisor/SKILL.md` for AI harnesses that use omnibias (not maintain it): quantity and order, structure,
  API family, optimizer, a `top_k=None` find, `explain()`, and the caveats to report. It links `docs/regimes.md` and copies no table. The agent
  context check allows user skills (`USER_SKILLS`) beside the package skills, and `AGENTS.md` routes users to it.
- Suite v2 was run (`omnibias_experiments/e2e_v2_20261024`, registered with Addendum A before any measurement and restarted from empty results under
  Amendment B): 265 pairs, 13 JAX problems from `D = 2` to `D = 10,000`, five seeds, one A100, float64. With compile excluded (the primary
  comparison of Addendum A, chosen after suite v1's outcome was known), 17 of 32 eligible problem-target rows have a lower 95 % bound of the
  ratio of at least 2 against the strongest baseline, on 7 of the 13 problems; in every one the candidate is a structured second-order pair
  (Gauss–Newton, SPRING, trust-region Levenberg–Marquardt or ENGD), and no Adam or L-BFGS candidate meets the bound. With the cold compile included 11
  rows meet it. Seven rows are losses or parity (`heat_hd_d1000` against STDE and SDGD, `triharmonic` against JAX jet, `logdet_flow_d16` against the
  dense Hessian), seven are not eligible, and no ranking exists at `poisson_d10000` or `hjb_d1000`. Of 19 registered predictions 9 hit, 7 missed
  and 3 were not scored. The record is `docs/benchmarks/planner/e2e_suite_v2_a100.json` (test: `tests/test_e2e_v2_record.py`); the section
  "End-to-end training" of `docs/operator-planner.md` replaces the statement that the suite had not been run, and the statement in
  `docs/performance.md` that it was not run now says it has since been run.
- Suite v2, float32 at `highest` precision (secondary): complete for 10 of the 13 problems (`heat1d`, `allen_cahn` and `cahn_hilliard` stopped
  part-way and could not be resumed, and no float32 row is reported for them); 8 of 22 eligible rows meet the bound with compile excluded, on four
  problems on which float64 does too, and the registered float32 prediction missed (113 of 192 pairs slower per step than float64).
- Full-scope matrix: the suite v2 rows are the new farm record `docs/benchmarks/farm_time_to_target_v2.json` (four comparison endpoints per
  problem and target, float64 and the ten complete float32 problems, a row with no ratio is incomplete and never a win), with the interval basis `t95_seeds` (the one-sided 95 % bound over
  five seeds, `t(0.95, 4)`, and its mirror `ratio^2 / lower`). The capability block of `.cursor/rules/omnibias.mdc` and `docs/regimes.md` are regenerated.
