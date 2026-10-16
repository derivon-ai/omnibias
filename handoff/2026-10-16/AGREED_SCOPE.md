# Agreed completion contract and outstanding validation

The exact 78-entry requirement registry, including evidence links and historical
statuses, is `completion_execution_20261013/completion_requirements_checklist_v15.json`
in the exported research tree. Read later source clarifications before treating
an older implementation-status sentence as current.

## Implementation scope

- Preserve package APIs/defaults/versions, previous reports and ledgers. Keep
  unqualified work private. Only exact final-package configurations passing
  correctness, conditional numerical qualification and performance gates can be
  exposed under `omnibias.jax.experimental`; no new root exports or mandatory
  Keras/Torch dependency.
- Explicit ridge, separable and affine-bottleneck descriptors; optional capture
  must establish structural assumptions. No arbitrary-Python support claim.
- Derivatives through16,17 internally for first parameter gradients; higher
  orders explicitly experimental. Recent steering prioritizes new improvements
  at orders2–6 without altering already frozen schedules.
- Provenance-safe expression/constant bindings; immutable numerical ownership;
  executable/certificate/selection identities separated; fingerprint once per
  process with explicit refresh; bounded16-plan LRU and optional persistent cache.
- Containing/repeated-domain reuse must preserve multiplicity and reduction/
  shared-parameter gradient bounds. Qualify actual scan/cond/slice/mask programs
  and emitted first gradients; unsupported operations refuse.
- Joint operator/objective/adjoint compilation, exact rational operator-span
  reduction, shared geometry/towers, live symbol/metric derivatives, zero-safe
  products, budgeted symbolic/state layouts and reconstruction roundoff.
- Objective-only lax.scan for explicit sums/weighted means/sufficient statistics;
  no concatenated outputs; two passes where needed; valid in-domain padded tails
  with masks and global normalization. Full-output requests keep output storage.
- Device-side validation, transactional parameters/optimizer state, active-flag
  step blocks, failure status/proposed update, and requalification before resume.
  Recenter radius=max(old radius,16*abs(update)); runtime domain bounds reusable
  only with valid certificates. Donation/checkpointing need ownership/gradient tests.
- One-worker/one-future double buffer; immutable dependency/batch tokens; frozen
  base/features may be prepared ahead; trainable projections cannot survive an
  update silently. Trace real overlap and charge host work/transfers/contention.
- Frozen-readout QR/SVD compression and qualified solves; explicit exact low-rank
  adapters A=A0+UV^T with full U,V gradients and cached arithmetic accounted for.
- Trainable-feature variable projection for fixed-target, weighted linear
  residuals: QR/triangular solves, explicit rank-aware SVD fallback, augmented
  user-requested regularization, residual/stationarity and verified positive
  singular-value bound. Never solve normal equations. Envelope first gradients
  include solve/feature/emitted errors; no readout Jacobian/higher-derivative claim.
  Streaming TSQR plus second feature pass stores O(H^2) factors plus chunks/state.
- Separate Fourier family (real/conjugacy/constant/Nyquist/dealias conventions),
  explicit bounded-twiddle radix2 multiplication/reciprocal and actual gradient
  analysis. Native FFT observational without an explicit backend error contract.
- Bounded diagnostic branches: polynomial lifting, approximate SVD and fast
  composition under unchanged contracts. Their failure/inconclusiveness is valid.

## Numerical contract

Forward atol=1e-9,rtol=2e-10; gradient atol=1e-8,rtol=2e-9. Do not loosen after
observing failures. Qualify outputs, scalar objective and every requested first
gradient of the actual emitted program. Native transcendental error and compiler
arithmetic hypotheses remain explicit; probes do not become vendor guarantees.
Binary64 round-to-nearest, gradual underflow and finite intermediates are stated
assumptions. Bounds crossing zeros must meet the absolute tolerance.

Retain256 complete candidates, at most4 measured specialization finalists,
4096 analysis cells and existing graph/state/process/evidence budgets. Refusal
is not a benchmark win. Do not infer rank/sparsity from a current numerical zero.

## Required scenarios

All families; shared/changed parameters; mixed and zero orders; complete small
Jacobians and unit/random VJPs; cancellation, saturation, derivative roots,
boundaries; known tanh16/32 nearz3 and8 nearz20; overflow and exhausted budgets.
Streamed/full gradients with uneven tails, metric adjoints and zero factors.
Provenance collisions, domain containment/invalidation/eviction. Accepted and
failed device steps, invalid batches, nonfinite gradients, first exits, exact
optimizer rollback and resume. Prefetch exhaustion/failure/stale dependencies.
Full-rank/deficient/zero/ill-conditioned/nearly-consistent LS and deliberately
inaccurate solves. UV gradients and unsupported base/input scopes. Fourier
conjugacy/Nyquist/dealiasing; FFT padding/normalization/reciprocal/cancellation.
Every newly advertised path needs nonvacuous value/objective/gradient bounds.

Prepared suite counts are separate/overlapping scopes, not a unique sum:
57 contract regressions,18 runtime-prepared cases,18 reusable-domain cases,
12 newer low-rank streaming backend methods, focused112/global68, structured17,
SCI16+saved8, MP17/stress36 and further saved/genuine-capture obligations.
Check the frozen runner definitions and latest source maps; these counts do not
mean they have executed. Historical CPU68/GPU68 do not validate new sources.

## Matched experiments

- Freeze separate calibration831001, validation831002 and measurement831003/
  fresh831004 inputs, sources and schedule before timing.
- Existing closed-form/sparse/generic/separableD4/D8/D16/bottleneck workloads at
  B32/4096; cache/stream/device/template-only and combined ablations; fixed and
  deterministic fresh batches; no-prefetch/double buffer; resident/streamed inputs.
- Readout reuse1/10/100/1000 constructs exactly the requested number of queries.
  Low-rank(D,H)=(64,32),(1024,64),ranks1/4/8. Equal Taylor QR/envelope VP controls,
  joint-training time-to-target separately. Series orders32/64/128/256.
- Canonical existing-compiler control must use each actual requested operator,
  not its historically inefficient supplied-operator fallback. Keep those old
  failures. Include native/analytic Taylor, fused manual and applicable folx,
  giving them identical structural/precompute/streaming/QR opportunities.
- CPU1,8 physical cores and GPU experiments separate. Three fresh serial
  processes per method/configuration, rotating order,3 warmups,51 synchronized
  samples. Process medians are statistical units; no replacement runs.
- Report true cold use, setup, plan/persistent cache hits, certificate updates,
  frozen invalidation, forward/VG, full training, validation, transfers and peak
  host/device memory separately. Account for copies and requalification.
- Paired log-speed ratios with one-sided95% Student-t lower bounds (df2 under
  timing assumptions). Lower>=1 in EVERY advertised regime for promotion;
  lower>=2 supports a significant-improvement claim. Not simultaneous/universal
  coverage. Missing/failed required members suppress the ratio.

## PDE requirements

Periodic [0,2pi). Heat u0=sin(x)+.5sin(2x), diffusivities.1/1,t_final=.5, exact
Fourier evolution. Burgers u0=sin(x),viscosities.1/.01,t_final=.3, independently
checked Cole–Hopf references and converged dealiased spectral controls. Targets
solution-relative L2 and normalized residual1e-3/1e-6, including initial and
periodic boundary errors. Freeze training before timing. Trainable-feature heat
VP is applicable; nonlinear-in-readout Burgers keeps its nonlinear solver.

## Completion sequence

Preserve original laptop schedule and failures. Once all its schedules end and
fresh idle authority is established on THAT machine, execute its original audits:

```
python audit_completion.py protocol.json --suffix _final
python audit_cache_lifecycle.py protocol_cache_lifecycle.json --suffix _final
python audit_pde_completion.py protocol_pde_completion.json
```

Do not run these against an incomplete exported prefix and call the original
schedule complete. Independently verify all science/gradients/timings/resources/
hashes. Current-source publication, numerical/genuine captures, qualified A100
calibration/matching, examples, full tables/failure ledger and earned integration
are outstanding. A new cluster schedule must be distinct from old laptop records.
