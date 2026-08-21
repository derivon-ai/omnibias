# Changelog

All notable changes to omnibias are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and each of the 42
distributions is versioned independently under semantic versioning.

## [Unreleased]

### Added — Jet-Hopfield (theory 09-13)

- `omnibias.core.jet_hopfield` plus architecture and hopfield twins:
  memories are germs; retrieve by contact mismatch.
- G1–G4 CI-gated. Temperature collapse only if `beta -> inf`. Not
  vector Hopfield. Not ImageNet. Not CCF stretch.
- Docs: `docs/api/jet_hopfield.md`,
  `docs/cookbook/jet-hopfield.md`.
  Smoke: `docs/benchmarks/jet_hopfield_smoke.json`.


### Added — Holonomic layer (theory 09-12)

- `omnibias.holonomic._core.layer`: the block is an Ore annihilator;
  forward prolongs the D-finite jet of `L u = 0`.
- G1–G4 CI-gated. D-finite class only. Not a general PINN. Not CCF
  stretch. Lean flags are 09-26, not asserted here.
- Docs: `docs/api/holonomic_layer.md`,
  `docs/cookbook/holonomic-layer.md`.
  Smoke: `docs/benchmarks/holonomic_layer_smoke.json`.


### Added — Collapse-Net (theory 09-11)

- `omnibias.core.collapse_net` plus
  `omnibias.{torch,jax}.architectures.collapse_net` and
  `omnibias.difference._core.collapse_net`: train a founding
  stencil, infer by `delta -> 0` collapse.
- G1–G4 CI-gated. Founding bias collapse only. Not a continuum
  PDE. Not CCF stretch.
- Docs: `docs/api/collapse_net.md`,
  `docs/cookbook/collapse-net.md`.
  Smoke: `docs/benchmarks/collapse_net_smoke.json`.


### Added — Riccati flow net (theory 09-10)

- `omnibias.core.riccati_flow` plus
  `omnibias.{torch,jax}.architectures.riccati_flow`: depth is the
  time-`t` flow of the founding Riccati ODE.
- G1–G4 CI-gated. Not a DEQ. Not a CNF. The flow itself is not
  bias collapse. Not CCF stretch.
- Docs: `docs/api/riccati_flow_net.md`,
  `docs/cookbook/riccati-flow-net.md`.
  Smoke: `docs/benchmarks/riccati_flow_net_smoke.json`.


### Added — sheaf-atlas net (theory 09-09)

- `omnibias.geometry.atlas.cocycle`: jet-valued transition maps
  with a cocycle residual ``Phi_kj ∘ Phi_ji - Phi_ki``.
- G1–G4 CI-gated. Chart jets are founding bias collapse
  (`delta -> 0`); partition hardening is temperature collapse
  (`beta -> inf`). Not a sheaf-cohomology theorem. Not P vs NP.
- Docs: `docs/api/sheaf_atlas_net.md`,
  `docs/cookbook/sheaf-atlas-net.md`.
  Smoke: `docs/benchmarks/sheaf_atlas_net_smoke.json`.


### Added — Characteristic-Net (theory 09-08)

- `omnibias.pinn.characteristic` plus
  `omnibias.pinn.{torch,jax}.characteristic`: transport along a
  learned `v` with a closed-form time integral and a shock flag.
- G1–G4 CI-gated. Jets of `v` are founding bias collapse
  (`delta -> 0`), not temperature collapse. Not 02-13. Not NS.
  Not a unique post-shock solution.
- Docs: `docs/api/characteristic_net.md`,
  `docs/cookbook/characteristic-net.md`.
  Smoke: `docs/benchmarks/characteristic_net_smoke.json`.


### Added — Frame-UNet (theory 09-04)

- `omnibias.core.frame_unet` plus
  `omnibias.{torch,jax}.architectures.frame_unet`: order encoder,
  integral decoder, band skip labelled apart from the collapse head.
- G1–G4 CI-gated. Jets are founding bias collapse (`delta -> 0`),
  not temperature collapse. `sigma'` is not admissible. Not ImageNet.
  Not CCF stretch.
- Docs: `docs/api/frame_unet.md`, `docs/cookbook/frame-unet.md`.
  Smoke: `docs/benchmarks/frame_unet_smoke.json`.


### Added — remainder training (theory 09-18)

- `omnibias.core.remainder_train` plus
  `omnibias.{torch,jax}.optim_remainder`: the loss is the Taylor
  remainder `R_N`, with an optional 03-13 birth hook.
- G1–G4 CI-gated. Jets are founding bias collapse (`delta -> 0`),
  not temperature collapse. Not spec 03-10. Not CCF stretch.
- Docs: `docs/api/remainder_training.md`,
  `docs/cookbook/remainder-training.md`.
  Smoke: `docs/benchmarks/remainder_training_smoke.json`.


### Added — Pack-MoE (theory 09-07)

- `omnibias.core.pack_moe` plus
  `omnibias.{torch,jax}.architectures.pack_moe`: experts are packs;
  the router is slab mass, not softmax.
- G1–G3 CI-gated. Jets are founding bias collapse (`delta -> 0`),
  not temperature collapse unless `beta != 1`. Not ImageNet. Not a
  05-02 LightGBM reversal. Not CCF stretch.
- Docs: `docs/api/pack_moe.md`, `docs/cookbook/pack-moe.md`.
  Smoke: `docs/benchmarks/pack_moe_smoke.json`.



### Added — coupling jet-flow (theory 09-06)

- `omnibias.core.coupling_flow` plus
  `omnibias.score.flow.{torch,jax}.jet_flow`: finite couplings whose
  log-det is `sum log|s| + sum log sigma'`, inverted by Newton.
- G1–G3 CI-gated. Jets are founding bias collapse (`delta -> 0`),
  not temperature collapse. Not `integrate_cnf`. Not ImageNet.
  Not CCF stretch.
- Docs: `docs/api/coupling_jet_flow.md`,
  `docs/cookbook/coupling-jet-flow.md`.
  Smoke: `docs/benchmarks/coupling_jet_flow_smoke.json`.



### Added — Taylor-model neuron and proof-carrying forward (theory 09-05 / 09-24)

- `omnibias.core.verified.tm_neuron` plus `omnibias.verify._core.pci`:
  a hidden state that is a `TaylorModel`, and a forward that
  returns `(y_mid, box)`.
- G1–G4 CI-gated. Jets are founding bias collapse (`delta -> 0`),
  not temperature collapse. Lean flags stay unforged. Not 08-09.
  Not a deep-net certificate. Not CCF stretch.
- Docs: `docs/api/tm_neuron.md`, `docs/api/pci.md`,
  `docs/cookbook/tm-neuron.md`.
  Smoke: `docs/benchmarks/taylor_model_neuron_smoke.json`,
  `docs/benchmarks/proof_carrying_forward_smoke.json`.



### Added — exact MAML (theory 09-16)

- `omnibias.core.exact_maml` plus `omnibias.{torch,jax}.optim_maml`:
  one inner Newton / GN step; meta-grad is IFT on inner
  stationarity (the chain rule).
- G1–G3 CI-gated. Jets are founding bias collapse (`delta -> 0`),
  not temperature collapse. Not ImageNet few-shot. Not CCF stretch.
- Docs: `docs/api/exact_maml.md`, `docs/cookbook/exact-maml.md`.
  Smoke: `docs/benchmarks/exact_maml_smoke.json`.



### Added — jet-token transformer and jet distillation (theory 09-02 / 09-19)

- `omnibias.core.jet_token` plus torch / jax `architectures.jet_token`
  and `jet_distill`: tokens are N-jets mixed by `compose_jet`; a
  student matches a teacher N-jet.
- G1–G3 CI-gated. Jets are founding bias collapse (`delta -> 0`),
  not temperature collapse. Not ImageNet, not Jet-KAN.
- Docs: `docs/api/jet_token.md`, `docs/cookbook/jet-token.md`.
  Smoke: `docs/benchmarks/jet_token_transformer_smoke.json`,
  `docs/benchmarks/jet_distillation_smoke.json`.


### Added — FTC-Net and dual-FTC training (theory 09-03 / 09-17)

- `omnibias.core.ftc` plus torch / jax `architectures.ftc_net`:
  the unused `integral` cell, its FTC derivative head, and a dual
  residual `(r_D, r_I)`.
- G1–G3 CI-gated. Jets are founding bias collapse (`delta -> 0`),
  not temperature collapse. Not a VPINN / weak form.
- Docs: `docs/api/ftc_net.md`, `docs/cookbook/ftc-net.md`.
  Smoke: `docs/benchmarks/ftc_net_smoke.json`,
  `docs/benchmarks/dual_ftc_training_smoke.json`.


### Added — Nobel-adjacent domain programs (theory 07-07)

- `omnibias.ferminet.hermite`, `omnibias.pinn.plasma`,
  `omnibias.pinn.stack`: an exact oscillator ladder, a
  Harris-sheet resistive-layer residual, and exact
  `dT/dtheta` through a transfer product.
- G0–G6 CI-gated as tooling. Jets are founding bias
  collapse (`delta -> 0`), not temperature collapse.
  Not a many-body solution, not fusion, not a new material.
- Docs: `docs/api/domain_programs.md`,
  `docs/cookbook/domain-programs.md`.
  Smoke: `docs/benchmarks/quantum_hermite_vmc_smoke.json`,
  `docs/benchmarks/plasma_resistive_layer_smoke.json`,
  `docs/benchmarks/materials_stack_design_smoke.json`.


### Added — validated dynamics and orbits (theory 07-06)

- `omnibias.core.verified.jet_flow`: exact-Jacobian
  `tower_jacobian`, order-`p` `lohner_step_jet`, a
  `WidthBudget` on every run, and a singularity-radius
  step suggestion. Existing `lohner_flow` is unchanged.
- G1–G6 CI-gated. Jets are founding bias collapse
  (`delta -> 0`), not temperature collapse. One field, one
  box, one finite horizon. Not a continuum existence
  theorem and not an attractor statement.
- Docs: `docs/api/validated_dynamics.md`,
  `docs/cookbook/validated-dynamics.md`.
  Smoke: `docs/benchmarks/validated_dynamics_smoke.json`.


### Added — spectral floors and adapted SOS bases (theory 07-05)

- `omnibias.core.verified.trial_spaces`: multi-pack trial
  spaces, alignment-required Temple / Lehmann floors, and
  residual birth / growth. `omnibias.sos.monomials` grows
  `arrangement_adapted_basis` so a named positivity set can
  certify at a lower ambient degree.
- G1–G6 CI-gated. Jets are founding bias collapse
  (`delta -> 0`), not temperature collapse. Fixed operator,
  one domain, one discretization. Not a continuum spectral
  gap and not a Yang-Mills mass gap.
- Docs: `docs/api/trial_spaces.md`,
  `docs/api/sos_adapted_basis.md`,
  `docs/cookbook/spectral-floors.md`.
  Smoke: `docs/benchmarks/spectral_trial_spaces_smoke.json`,
  `docs/benchmarks/sos_adapted_basis_smoke.json`.


### Added — weak-form NS-adjacent enclosures (theory 07-02)

- `omnibias.pinn.certified.weak_form`: a width decomposition
  (repr / quad / deriv / round) and a weak-form Taylor-Green
  enclosure whose quadrature term is 2x narrower because the
  Gauss piece is gone. Exact-jet Lohner vs a polluted Jacobian.
- G1–G6 CI-gated. Jets are founding bias collapse
  (`delta -> 0`), not temperature collapse. Finite box,
  finite horizon, finite test space. Not a continuum
  Navier-Stokes regularity claim.
- Docs: `docs/api/ns_weak_form.md`,
  `docs/cookbook/ns-weak-form.md`.
  Smoke: `docs/benchmarks/ns_weak_form_enclosure_smoke.json`.

### Added — Lie symmetry discovery (theory 03-11)

- `omnibias.symbolic.symmetry`: determining-matrix nullspace
  for Lie point symmetries in a declared affine ansatz.
  Prolongation is the characteristic / total-derivative jet
  algebra. Rank threshold and singular-value separation are
  required output.
- G1–G6 CI-gated. Jets are founding bias collapse
  (`delta -> 0`), not temperature collapse. Point symmetries
  only; in-ansatz dimensions, not a classification.
  Finite-difference prolongation gets the heat rank wrong.
- Docs: `docs/api/symmetry_discovery.md`,
  `docs/cookbook/symmetry-discovery.md`.
  Smoke: `docs/benchmarks/symmetry_discovery_smoke.json`.

### Added — jet-Padé singularity tracking (theory 03-10)

- `omnibias.difference.singularity` plus
  `omnibias.fields.singularity`: Domb-Sykes and Padé poles
  locate the nearest singularity; a coefficient tail bound
  encloses `|x_s|`. `SingularityTrack` always carries a
  blow-up disclaimer.
- G1–G6 CI-gated. Jets are founding bias collapse
  (`delta -> 0`), not temperature collapse. Diagnostic and
  estimate, not a proof of blow-up. Essential singularities
  report failure. Froissart filter is machine-epsilon, not
  suite-tuned.
- Docs: `docs/api/singularity_tracking.md`,
  `docs/cookbook/singularity-tracking.md`.
  Smoke: `docs/benchmarks/singularity_tracking_smoke.json`.

### Added — differentiable topology (theory 03-09)

- `omnibias.shape.topology`: soft Euler characteristic and
  soft component counts with a certified spectral integer
  route, plus 1-D Morse persistence and a persistence loss.
- G1–G6 CI-gated. No differentiable function equals a Betti
  number. `beta -> inf` is temperature collapse, not founding
  bias collapse. `Inconclusive` when the spectral gap does
  not separate. Persistence is differentiable almost
  everywhere.
- Docs: `docs/api/differentiable_topology.md`,
  `docs/cookbook/differentiable-topology.md`.
  Smoke: `docs/benchmarks/differentiable_topology_smoke.json`.

### Added — certified scan localization (theory 03-08)

- `omnibias.verify.localization`: Krawczyk unique-peak
  enclosure of a closed-form scan response, with
  `Inconclusive` as a first-class refusal and a hash-sealed
  v1 certificate.
- G1–G6 CI-gated. The template is founding bias collapse
  (`delta -> 0`), not temperature collapse. Scope is
  `local_box`. Not `theorem_prover_verified`. A deterministic
  enclosure of noisy data is conditional on the data.
- Docs: `docs/api/certified_localization.md`,
  `docs/cookbook/certified-localization.md`.
  Smoke: `docs/benchmarks/certified_localization_smoke.json`.

### Added — scale flow and coarse-graining (theory 03-07)

- `omnibias.core.scale` plus `omnibias.fields.scale`: exact
  `alpha^n` rescaling, closed-form gaussian overlaps, linear
  Galerkin coarse-graining, a derived band schedule, and a
  grid-free V-cycle. Nonlinear `FlowSystem` records the
  truncation order and refuses exponents without it.
- G1–G6 CI-gated. `alpha` is a tempering scale, not founding
  bias collapse and not temperature collapse. Linear
  coarse-graining is exact; nonlinear flow is truncated.
- Docs: `docs/api/scale_flow.md`, `docs/cookbook/scale-flow.md`.
  Smoke: `docs/benchmarks/scale_flow_smoke.json`.

### Added — neural quadrature (theory 03-06)

- `omnibias.core.cubature`: moment-solved Gauss / pack quadrature
  with a Peano `certified_error` enclosure, plus
  `integrate_certified` on the fields quadrature surface.
- G1–G5 CI-gated. Pack moments are founding bias collapse
  (`delta -> 0`), not temperature collapse. The enclosure is
  refused without a bound on `f^(d+1)`. Non-product cubature
  is out of scope.
- Docs: `docs/api/neural_quadrature.md`,
  `docs/cookbook/neural-quadrature.md`.
  Smoke: `docs/benchmarks/neural_quadrature_smoke.json`.

### Added — differentiable morphology (theory 03-05)

- `omnibias.shape.morphology`: soft dilation / erosion via
  `logsumexp_beta`, composition-aware `log(N)/beta` gap,
  learnable pack structuring elements, soft max-pool, and a
  soft Euclidean distance transform.
- G1–G6 CI-gated. Soft max is temperature collapse
  (`beta -> inf`, feasibility), not founding bias collapse
  (`delta -> 0`). The gap is worst-case. Not a seventh
  `OperatorBlock` role.
- Docs: `docs/api/morphology.md`, `docs/cookbook/morphology.md`.
  Smoke: `docs/benchmarks/morphology_smoke.json`.

### Added — sliced optimal transport (theory 03-04)

- `omnibias.measure.transport`: exact 1-D `W_1` between
  tempered-activation mixtures (sign-change roots + softplus
  antiderivatives) and a sliced average that always reports
  `direction_stderr`.
- G1–G6 CI-gated. Founding bias collapse (`delta -> 0`), not
  temperature collapse. Exact per slice, not sample-free. Sliced
  Wasserstein is not Wasserstein.
- Docs: `docs/api/sliced_ot.md`, `docs/cookbook/sliced-ot.md`.
  Smoke: `docs/benchmarks/sliced_ot_smoke.json`.


### Added — constraint satisfaction collapse (theory 03-03)

- `omnibias.discrete.csp`: finite-domain CSP on the
  `DiscreteProblem` seam. Multilinear relations, compact
  all-different / cardinality relaxations, two independent
  temperature schedules, and soft arc consistency.
- G1–G6 CI-gated. Both `beta` knobs are temperature collapse
  (`beta -> inf`), not founding bias collapse (`delta -> 0`).
  Not a complete solver. Not P vs NP.
- Docs: `docs/api/csp.md`, `docs/cookbook/csp-collapse.md`.
  Smoke: `docs/benchmarks/csp_collapse_smoke.json`.


### Added — arrangement LP / learned facets (theory 03-02)

- `omnibias.convex.arrangement`: a learned-facet front end on
  `solve_lp` and the Neumaier-Shcherbina `lp_dual_lower_bound`.
  Soft cell membership is temperature collapse; KKT is exact
  where the active set is stable. Vertex enumeration is a
  small-instance check only (`D<=4`, `n<=16`).
- G1–G5 CI-gated. Not a new LP algorithm. No uncorrected float
  dual path. Not P vs NP.
- Docs: `docs/api/arrangement_lp.md`,
  `docs/cookbook/arrangement-lp.md`.
  Smoke: `docs/benchmarks/arrangement_lp_smoke.json`.

### Added — soft-population evolution (theory 03-01)

- `omnibias.discrete.evolution`: softmax selection
  `w = softmax(-beta E)` with a closed-form `log(P)/beta`
  gap, geometry-aware pack mutation, exact-Newton memetic
  polish (full evaluation accounting), and a certify-loop
  stop on `DiscreteProblem`. Torch / jax twins specialise
  `soft_weights`; evolve loops use `numpy.random.Generator(seed)`.
- G1–G6 CI-gated. Selection is temperature collapse
  (`beta -> inf`), not founding bias collapse
  (`delta -> 0`). Not a P = NP claim, and G2 targets
  parity with CMA-ES rather than beating it.
- Docs: `docs/api/soft_evolution.md`,
  `docs/cookbook/soft-evolution.md`.
  Smoke: `docs/benchmarks/soft_evolution_smoke.json`.

### Added — rational stencil Lean obligations (theory 01-11)

- `omnibias.core.proof.obligations.rational_stencil`: seal
  `C_j` consistency and Birkhoff poisedness as finite rational
  identities. The Mathlib-free kernel decides them with
  `allRatEq` / `allIntGe` / `ratNez`.
  `theorem_prover_verified` is earned only by a genuine
  `lake build`. `mathlib_verified` stays false.
- G1–G5 CI-gated (kernel pass in the Lean job). Lean
  certifies the algebra, not the collapse, not Taylor's
  theorem, and not a function-class remainder.
- Docs: `docs/api/rational_stencil.md`,
  `docs/cookbook/rational-stencil.md`.
  Smoke: `docs/benchmarks/rational_stencil_smoke.json`.

### Added — certified step (theory 08-09)

- `omnibias.verify.train_step`: accept a trial `theta'` only when a
  sealed Lipschitz or output-box enclosure stays inside a declared
  cap. Empty or overflowing enclosures return `reason="vacuous"`,
  never `accepted=True`. Not imported by T1 `omnibias.{torch,jax}`.
- G1–G3 CI-gated. Distinct from 08-04 (root of `F`). Not a global
  min, not CCF stretch, and not a robustness claim without an
  enclosure. Bias collapse (`delta -> 0`) supplies `sigma'`.
- Docs: `docs/api/certified_step.md`,
  `docs/cookbook/certified-step.md`.
  Smoke: `docs/benchmarks/certified_step_smoke.json`.

### Added — implicit DEQ Newton (theory 08-08)

- `omnibias.core.implicit` plus `omnibias.{torch,jax}.implicit` twins:
  solve `u = sigma(W u + x)` by Newton or Banach iteration and
  differentiate the root with exact `sigma'` (IFT VJP). JAX uses
  `lax.while_loop` for the default loop. Anderson is extra and raises.
  `require_contraction=True` raises rather than silently unrolling.
- G1–G3 CI-gated. IFT is the chain rule at a fixed point, not unrolled
  BPTT, not a global min, and not CCF stretch. Bias collapse
  (`delta -> 0`) supplies `sigma'`.
- Docs: `docs/api/implicit.md`, `docs/cookbook/implicit-deq.md`.
  Smoke: `docs/benchmarks/implicit_deq_smoke.json`.

### Added — depth-causal residual (theory 08-05)

- `omnibias.pinn.train._core.depth_residual` plus
  `omnibias.pinn.train.{torch,jax}.depth_residual` twins: after each
  hidden layer a linear decode is a field, the PDE residual is formed
  from a closed-form `layer_jet` / `jet_to_tower`, and a damped
  Gauss–Newton step updates only that layer. Not a rewrite of time
  marching. Distinct from 08-03 (proxy residual).
- G1–G3 CI-gated. A greedy depth march, not a global min, not CCF
  stretch, and not Hilbert. Bias collapse (`delta -> 0`) supplies the
  tower. Honesty keys are sealed
  (`navier_stokes_proof_claim` / `stretch_1e-13_cleared` false,
  `hilbert_not_in_scope` true).
- Docs: `docs/api/depth_residual.md`,
  `docs/cookbook/depth-causal-residual.md`.
  Smoke: `docs/benchmarks/depth_causal_residual_smoke.json`.


### Added — depth-causal local jet (theory 08-03)

- `omnibias.core.local_jet` plus `omnibias.{torch,jax}.train_local`
  twins: a reverse-depth local Gauss–Newton step on a named residual
  (readout / invert-and-match / predictive coding) with a compressed
  `k`-direction `layer_jet`. `n_directions >= n_params` raises
  `LocalJetForbidden` unless `allow_full=True`. Not re-exported from
  `omnibias.{torch,jax}.optim` (avoids an import cycle).
- G1–G4 CI-gated. A greedy warm start, not a global min, not ImageNet,
  and not CCF stretch. Bias collapse (`delta -> 0`) supplies the tower.
  Not a rewrite of `omnibias.pinn.train`.
- Docs: `docs/api/local_jet.md`,
  `docs/cookbook/depth-causal-local-jet.md`.
  Smoke: `docs/benchmarks/depth_causal_local_jet_smoke.json`.

### Added — block / coordinate exact search (theory 08-07)

- `omnibias.core.block_search` plus
  `omnibias.{torch,jax}.optim_block_search` twins: 03-12 jet line
  search on a last-linear / OMBU-bias / arrangement-`W` block with
  `verify=True`. Last-layer least squares is exactly quadratic.
  Re-exported from `omnibias.{torch,jax}.optim`.
- G1–G4 CI-gated. A coordinate sweep, not a global solver and not CCF
  stretch. Bias collapse (`delta -> 0`) supplies the tower.
- Docs: `docs/api/block_exact_search.md`,
  `docs/cookbook/block-exact-search.md`.
  Smoke: `docs/benchmarks/block_exact_search_smoke.json`.

### Added — sharpness-scheduled curvature step (theory 08-06)

- `omnibias.core.sharpness` plus
  `omnibias.{torch,jax}.optim_sharpness` twins: exact-HVP Lanczos
  `lambda_max` sets cubic `sigma` (or a learning rate) each step.
  Hooked on `CubicNewton` / `CubicRegularizedNewton` via
  `sharpness=`. A non-positive schedule refuses the step. Re-exported
  from `omnibias.{torch,jax}.optim`.
- G1–G3 CI-gated. Sharpness is a step-size signal, not a
  generalization claim and not CCF stretch. Hutchinson is not the
  method. Bias collapse (`delta -> 0`) makes HVPs exact.
- Docs: `docs/api/sharpness_schedule.md`,
  `docs/cookbook/sharpness-schedule.md`.
  Smoke: `docs/benchmarks/sharpness_schedule_smoke.json`.

### Added — Kantorovich-accepted Newton (theory 08-04)

- `kantorovich_accept_step` on
  `omnibias.core.verified.kantorovich` plus
  `omnibias.{torch,jax}.optim_kantorovich` twins: a Gauss–Newton trial
  is kept only when the radii polynomial returns a nonempty unique-zero
  ball of a finite residual map. Empty ball is a valid reject.
  `continuum_pde_claim` is sealed false; `theorem_prover_verified` is
  not asserted. Re-exported from `omnibias.{torch,jax}.optim`.
- G1–G3 CI-gated. Not a continuum PDE theorem, not CCF stretch, and not
  a Lean kernel pass. Bias collapse (`delta -> 0`) is not required.
- Docs: `docs/api/kantorovich_newton.md`,
  `docs/cookbook/kantorovich-newton.md`.
  Smoke: `docs/benchmarks/kantorovich_newton_smoke.json`.

### Added — composed-curvature joint Newton (theory 08-02)

- `omnibias.core.composed_curvature` plus
  `omnibias.{torch,jax}.optim_composed` twins: order-2 chain rule
  Hessian of consecutive layers, slice-vs-joint eigenvalues, damped
  joint Newton or a negative-mode escape. Length uses the 03-12 jet
  line search (`verify=True`). `n_directions >= n_params` raises unless
  `allow_full=True`. Re-exported from `omnibias.{torch,jax}.optim`.
- G1–G4 CI-gated. Escape is from a slice critical point on a local
  Poisson residual, not a global min and not CCF stretch. `sigma''`
  comes from bias collapse (`delta -> 0`).
- Docs: `docs/api/composed_curvature.md`,
  `docs/cookbook/composed-curvature.md`.
  Smoke: `docs/benchmarks/composed_curvature_smoke.json`.

### Added — adaptive pack refinement (theory 03-13)

- `omnibias.core.refine` plus bit-identical
  `omnibias.{torch,jax}.refine` twins: birth (`c = 0`) and collapsed
  *p*-type growth are bit-identical; death reports
  `death_perturbation`. `GrowableOMBU.grow` remains the literal-`K`
  path (torch). Scale `alpha` is the founding scaling law, not
  temperature collapse. Singularity / scale-flow indicators use
  Domb-Sykes and jet ratios, not the full 03-10 / 03-07 modules.
- G1/G2/G3/G5/G6 CI-gated. G4 (10x vs a fixed bank) earned on the
  smoke lstsq reconstruction and is recorded, not in CI `all_passed`.
- Docs: `docs/api/refine.md`, `docs/cookbook/adaptive-pack-refinement.md`.
  Smoke: `docs/benchmarks/adaptive_refinement_smoke.json`.

### Added — exact jet line search (theory 03-12)

- `omnibias.core.line_search` plus bit-identical
  `omnibias.{torch,jax}.line_search` twins: directional Taylor model,
  certified Lagrange truncation radius when a `|phi^(N+1)|` bound is
  supplied, Wolfe-as-interval, interval-Newton root isolation, and a
  `verify=True` never-worse backstop. Input-ray path reuses `mlp_jet`
  and `compose_jet`. Re-exported from `omnibias.{torch,jax}.optim`.
- G1/G2/G3/G6 CI-gated. G4 (step-count vs Wolfe) and G5 (cost-crossover
  table) are recorded, not in CI `all_passed`. The polynomial is a
  model, not the loss. `taylor_line_min` is unchanged. CCF stretch and
  a global min of a deep nest stay unearned.
- Docs: `docs/api/line_search.md`, `docs/cookbook/jet-line-search.md`.
  Smoke: `docs/benchmarks/jet_line_search_smoke.json`.

### Changed — Phase-0 reproduce Hilbert is `wholeline_hp`

- `reproduce_deepmind_config` / `benchmarks/reproduce_deepmind_ccf.py`
  train Hilbert is `wholeline_hp` (split-core GL + power-mapped tail)
  with `proj_defect_weight=0`. Periodic truncated-line FFT and
  finite-interval `hilbert_pv_line` stay labeled numerical diagnostics
  (they err at `O(1e-1)` vs `H[Q]=-P`). `hardy_corrected_pv` uses
  `hilbert_wholeline_hp` on the remainder when `omega_fn` is set; PV is
  the no-`omega_fn` fallback. Default reproduce device is CPU.
- CI table: FFT/PV `O(1e-1)` vs hp `≤ 1e-8` on a leading Hardy atom;
  unprojected neural Ω vs a higher-order hp reference `≪ 1e-4`;
  torch/JAX hp parity. Stretch `1e-13` and
  `navier_stokes_proof_claim` stay unearned. Earn-path default remains
  `hardy_projection`.

### Added — Jacobian n=2 counterexample box

- `omnibias.holonomic.jacobian_n2` names the finite universal
  `C_box(d, h, G)`: every integer-coefficient map `Q^2 -> Q^2` of
  degree `<= d` and height `<= h` either has `det JF` not identically a
  nonzero constant, or has no two points of a rational grid `G` sharing
  an image. Polarity is universal. A hit would be a genuine `n=2`
  counterexample. A miss is not injectivity on `Q^2` and not the parent.
- `identical_jacobian_constant` is an identity in `Q[x]`, not a probe
  sample. The n=3 tangent-sweep gate now uses it (`1+z` agrees on the
  plane `z=0` and is not constant). Breakthrough: axis probes can lie.
- CI exhausts the complete degree-1 / height-1 affine box (729 maps)
  and the degree-0 box (9 maps). Both prove `C_box` only.
  `escalate_n2_result` sets `jacobian_n2_claim` only on an earned
  violator. `jacobian_conjecture_proof_claim` stays False.
- Catalog kind `jacobian_n2_degree_box`. Degree `>= 2` is an incomplete
  structured slice. Cookbook: `docs/cookbook/jacobian-n2-box.md`.
- Gabber inverse-degree test: a Keller map whose unique origin-centered
  inverse jet of degree `<= deg F` is not a polynomial inverse is an
  exact `n=2` violator (`gabber_n2_test`). Catalog kind
  `jacobian_n2_homogeneous` is the `I +` homogeneous integer box.
  A miss is still not the parent.

### Added — autonomy engineering stack

- `omnibias.symbolic.ingest` packs jets / design / sequence / graph /
  poly / residual into a tag-optional `Observation`. Binders read the
  table (`sample_x`, column names, `poly_constraints`), not planted
  tags. `observation_square` / `observation_abs` are packed tables.
- `select_class(..., grow=True)` applies one legal constructor step
  (or `LinearSpanFamily` column products) after an exact miss. The
  grammar stays incomplete. `budget == 0` does not grow.
- New registered sorts: `dfinite`, `sos_onset`, `edge_colouring`,
  `extremal_template`, `residual_sign`. Factories stay human-authored.
- `run_bilevel_class_loop` proposers: `stlsq`, `neural_jet`, optional
  `fit_field`, optional `pinn_train` (pinn is not imported by default
  and is off in CI). Snap remains the accept gate.
- `ClassMemory.save` / `load` optional JSON. `Observation.features()`
  is length-`FEATURE_DIM` and includes table flags. `LogitGate` follows
  that width.
- `load_discovery_stack` / `discover_observation` is the one entry
  point. Core still does not import holonomic / sos / combinatorics /
  pinn.

### Added — shared-observation class loop

- `Observation` is a tagged, JSON-able payload (sequence / jet / design /
  graph / polynomial). `bind_sorts` applies registered
  `Observation → FiniteFamily | None` factories. A dropped binder
  forces an incomplete grammar. Planted defaults fire only when the
  tag or payload matches, or when no observation is supplied.
- `select_class` is the front door: bind → optional gate order →
  `run_discovery(..., collect=True)` → `rank_class_hits`. Exact snap /
  finite predicate outranks enclosure, which outranks empirical. Soft
  RMSE never outranks an exact identity. The driver never sets
  `GrammarSpec.complete=True`. `no_condition_exists_claim` stays False.
- `ClassMemory` / `FrequencyGate` (core) and `LogitGate` (numpy, with
  torch / jax logit twins) only propose a sort order. They never write
  `ExactCheck`. `run_bilevel_class_loop` runs STLSQ then snap on a
  design/target observation, then `select_class`.
- `LinearSpanFamily` is a bitmask nullity-one checker over an exact-`Q`
  table carried on the observation, so jet / PDE / conservation binders
  are views of one table.

### Added — condition-language discovery

- `omnibias.core.proof.condition` is a hypothesis language whose points
  are typed conditions (`ConditionHypothesis` / `GrammarSpec`), not
  colourings or maps. Typed constructors (`compose_jets`,
  `multiply_columns`, `raise_ore_degree`, `split_partition`,
  `add_minor_edge`) grow the grammar. Depth overflow and untyped
  emission return `None`.
- `KindMetaFamily` tries registered condition sorts. Missing sorts
  force an incomplete grammar. An incomplete miss is
  `search_incomplete`. A complete exhausted miss is `BLOCKED` with
  `no witness in enumerated grammar`. `no_condition_exists_claim` stays
  False.
- `omnibias.core.proof.lift.residual_identically_zero` is the exact-`Q`
  accept gate. `snap_sparse_equation` calls it. NeuralJet / STLSQ /
  PINN residuals stay `empirical` until they snap
  (`omnibias.symbolic.propose`).
- Cheap exact families: jet / PDE / conservation / integer-order /
  piecewise (`omnibias-symbolic`), Ore (`omnibias-holonomic`), SOS
  template (`omnibias-sos`), named minors on `n≤5`
  (`omnibias-combinatorics`). Catalog kinds `condition_*`.

### Added — discovery engine (catalog, lift, characterization)

- `omnibias.core.proof.catalog` registers discoverable kinds without
  core importing downstream packages. Modes are `exact_search`,
  `exact_replay`, `enclosure`, and `empirical`. Soft residuals never
  become an `ExactCheck`.
- `run_discovery` now tracks exhaustion (`cardinality` + iterator
  end). `budget == 0` is always `search_incomplete`. Universal
  statements can `DISPROVE` (counterexample) or `PROVE` a finite
  universal on an exhausted complete family. `collect=True` returns a
  solution set. `DiscoveredEquation` / `Characterization` are
  box-scoped; first-hit `PROVED` is not uniqueness.
- Exact span families: recurrence / tanh identity / planted heat
  (`omnibias-symbolic`), prefix-verified holonomic guesses, blind
  `K_5` colouring search, extremal template search, SOS degree box.
  Optional `AnnealDescentProposer` lives in `omnibias-discrete` and is
  not a core `get_proposer` name.

### Added — discovery-machine loop (statement → family → proposer → exact check)

- `omnibias.core.proof.discovery` is the proposer layer on
  `ProofMachine`: `Statement`, `FiniteFamily`, `run_discovery`. CI
  default is `score_guided`. `coordinate_newton` is a discrete
  finite-difference walk (not `CubicNewton`). `onehot_anneal` is a
  1-flip box anneal (not `omnibias-qubo`). A hit is a family witness.
  A miss on an incomplete family is `BLOCKED`.
- Holonomic `keller_tangent_sweep_deg3` (height ≤4) and combinatorics
  `dgg_dag_le6` (capped 3-terminal DAGs). The DAG CI default may miss.
  Planted 6-vertex recovery is required. H* remains the box CI must
  hit. Parent honesty flags stay `False`.

### Added — finite prove/disprove gates (Keller, DGG, Ramsey, extremal)

- `omnibias.holonomic.keller` replays Alpöge / Gallagher as exact `Q`
  Jacobian identities and runs a blind deg-2 tangent-sweep search.
  `build_holonomic_machine()` registers `keller_alpoge_replay` and
  `keller_tangent_sweep`. `PROVED` is the finite obligation, not a
  Jacobian-conjecture proof; `n=2` stays open.
- `omnibias.combinatorics.unsplittable` replays the AFP / Rybin H*
  instance (fractional 58 vs legal unsplittable 60) and searches the
  H* parameter box. The 1999 congestion theorem stays true. Finding a
  separator is a family stress-test, not "omnibias disproved Goemans."
- `omnibias.combinatorics.ramsey` checks triangle-free colourings
  (`R(3,3)>5` smoke) and a tiny `IsSaturated` matrix.
  `erdos_183_claim` is never `True`.
- `omnibias.combinatorics.extremal` encodes `C4`/`C6`/`jTemplate`/
  `kTemplate`/`pairGraph(4,2)` and checks finite predicates.
  `erdos_146_claim` / `erdos_180_claim` are never `True`.
- `build_combinatorics_machine()` registers the replay and search
  kinds. Cookbooks: `keller-jacobian`, `unsplittable-flow`,
  `finite-ramsey-colouring`, `extremal-graph-replay`.

### Added — DeepMind-gap absorbs (stage-2 Wang GN, Boussinesq q/β, PirateNet)

- JAX stage-2 ``optimizer="martens_grosse"`` and torch
  ``optimizer="wang_linearized_gn"`` run residual-vector Gauss–Newton on
  paper eq. 19 ``R0 + ε D[Φ0] Φ1``. The numpy torch path stays the labeled
  ``gauss_newton_corr_proxy``. Does **not** earn CCF stretch ``1e-13``.
- ``omnibias.pinn.{jax,torch}.equations.boussinesq_compactified`` lifts
  hats through the paper ``(q, β)`` chart with closed-form envelope jets.
  Optional ``BoussinesqDiscoveryConfig(compactified=True)``. Scaffold
  only; not a DeepMind residual or Navier–Stokes claim.
- Reusable ``omnibias.{jax,torch}.architectures.piratenet`` α-skip
  (``α=0`` is identity). CCF ``pirate_hat`` consumes it. Not ImageNet /
  ViT, not stretch.
- `deepmind_paper_architecture_config` / `deepmind_signed_hat_config`
  select the Wang et al. stack (compactified tanh MLP, optional
  exp-adjacent hat, gradient-normalized residual, Martens–Grosse,
  `wholeline_hp`) without silently flipping
  `reproduce_deepmind_config.exp_core`. `deepmind_multistage_config` +
  `stage2_even_hat` are the paper MSNN stage-2 on even compactified
  `q` (CCF hat stays even). A same-Jacobian bake-off on even
  Fourier stage-2 ranked epigraph L∞ first for the 1601-pt L∞
  stretch metric; L2 / peak-weighted / IRLS raised predicted
  L∞. `train_stage2` now uses even compactified `q` when
  `even_q=True` (DeepMind CCF path); `identity_readout` is
  exact Φ1=0; `gradient_normalize_residual` is the follow-up
  reweight. A later identity-init even-`q` MSNN bake-off
  measured `max|r| = 7.943e-3`; a second identity-init even-`q`
  MSNN (new seed) later measured `7.938e-3`. Even Chebyshev
  \(T_k(2q-1)\) on compactified `q` later measured
  `7.889e-3`. Even Legendre \(P_k(2q-1)\) later measured
  `7.885e-3`. Even Padé `[4/4]` on \(x=2q-1\) later
  measured `7.884e-3` (den Jacobian dead at `num=0`).
  Higher even Fourier \(k=6..12\) later measured `7.876e-3`.
  An identity-init tanh MLP readout on even `(q, ξ, bump)`
  later measured `7.847e-3`. Even compact Gaussians at the
  origin and residual peak later measured `7.840e-3`.
  An identity-init SiLU MLP readout on even `(q, ξ, bump)`
  did not promote (LP pred earn `~1.4e-7`; line search
  raised L∞).   Unfreezing the tanh hidden as additive
  `ΔW`/`Δb` did not promote (LP pred earn `~1.74e-6`;
  line search raised L∞). A multiplicative even Fourier
  on \(\xi=y^2/(1+y^2)\) did not promote (LP pred earn
  `~8.2e-5`; line search raised L∞). Identity-init GELU
  on \((\rho, E, \rho-q)\) did not promote (LP pred earn
  `~3.4e-7`); paper L2 moved `HΩ(0)` toward `+1.30` but
  raised L∞. Identity-init paper MSNN on
  \(\eta=[(2/\pi)\arctan y]^2\) did not promote (LP pred
  earn `~1.87e-5`; improving steps walked the peak to
  `|y|≈38`). Identity-init Mish on a log1p chart did not
  promote (tail-capped LP pred earn `~1.59e-6`; line
  search raised L∞). Even \(\operatorname{sinc}(y/s)\)
  realized a true L∞ earn of `~7.0e-7` (below the
  `1e-6` promote gate). Even cylindrical \(J_0(y/s)\)
  did not promote (LP pred earn `~7.5e-7`; far-field
  peak walk). A residual-shaped even lift realized only
  `~3e-11`. Unfreezing the PirateNet last layer did not
  promote (LP pred earn `~7.4e-6`; line search raised
  L∞). Even Hermite–Gauss functions did not promote
  (LP pred earn `~3.6e-5`; line search raised L∞).
  Unfreezing stage-3 MSNN Fourier frequencies did not
  promote (LP pred earn `~3.4e-6`; line search raised
  L∞).   Unfreezing the PirateNet skip-gate `α` did not
  promote (LP pred earn `~8e-9`). Identity-init tanh
  on an asinh chart did not promote (LP pred earn
  `~1.7e-6`; line search raised L∞). Even
  \(y\,\mathrm{dawsn}(y/s)\) did not promote (LP pred
  earn `~2.1e-7`; L∞ steps walked to `|y|≈38`). Unfreezing
  PirateNet embedding bias `Δbe` did not promote (LP pred
  earn `~2.6e-6`); an exact Hilbert-quadratic ray along
  that direction had no in-field improving scale.
  Even \(\tanh(\sinh(y/s))^2\) plus the same
  Hilbert-quadratic ray did not promote (LP pred
  earn `~1.6e-7`; model `s=0`; far-field peak walk).
  Later decaying special-function ticks (Airy,
  modified Struve, Kummer
  \({}_1F_1(3/2;5/2;-(y/s)^2)\), Whittaker
  \(e^{-z/2}U(1,1,z)\)) stayed in-field but
  realized only `~2e-11`–`~5e-11`.
  Paper L2 / grad-norm / exp-mult raised L∞
  at this basin. Stretch stays unearned.

### Fixed — official free-Ω path stays float64

- `hilbert_wholeline_hp` / `free_omega_vorticity_residual` require
  `jax_enable_x64` (JAX's default is float32). GL nodes, lambda
  compactification, the decay envelope, and `hard_gauge` now pin
  `float64` so the 1601-pt Wang score cannot silently demote.

### Added — signed PirateNet hat (jaxpi block, official envelope)

- `omnibias.pinn.jax.discovery.pirate_hat` ports the jaxpi PirateNet
  identity-init residual-adaptive block onto the official Wang lift
  `Ω = y · E · hat` with a *signed* hat (no softplus). Official
  `CompactifiedOmegaOMBU` defaults `exp_core=True` and cannot represent
  the signed champ. Even coordinates are `(q, y²/(1+y²))` so the
  near-field is not crushed by compactified `q` alone. `fit_pi_init`
  least-squares the linear readout; `fit_hat_l2` is a Hilbert-free
  Gauss–Newton hat match. Residual scoring stays
  `free_omega_vorticity_residual`. Prototype. Stretch `1e-13` and
  `navier_stokes_proof_claim` stay unearned. Rung-1 is not started
  here. jaxpi / fluid-singularities / Unstable-Singularity-Detector
  were audited and are not installed as the CCF solver (none ships a
  working CCF `1e-13` path). A zero-readout correction on the
  frozen erf-sinh stack later measured official-path
  `max|r| = 8.063e-3` (`HΩ(0)≈+1.003`; erf-sinh parent was
  `8.068e-3`). An even Fourier stage-2 on compactified `q`
  (`kπ`, `k=1..5`) with epigraph L∞ then one refine later
  measured `max|r| = 7.991e-3` (`HΩ(0)≈+1.004`). Identity-init
  even-`q` MSNN (paper small tanh Fourier, additive) with
  epigraph L∞ then one refine later measured
  `max|r| = 7.943e-3` (`HΩ(0)≈+1.005`). A second identity-init
  even-`q` MSNN (new seed) with epigraph L∞ then one refine
  later measured `max|r| = 7.938e-3`. Even Chebyshev
  \(T_k(2q-1)\) on compactified `q` with epigraph L∞
  then one refine later measured `max|r| = 7.889e-3`
  (`HΩ(0)≈+1.006`). Even Legendre \(P_k(2q-1)\)
  with epigraph L∞ later measured `max|r| = 7.885e-3`.
  Even Padé `[4/4]` on \(x=2q-1\) with epigraph L∞
  then one refine later measured `max|r| = 7.884e-3`
  (denominator columns vanished at `num=0`).
  Higher even Fourier \(k=6..12\) with epigraph L∞ later
  measured `max|r| = 7.876e-3`.
  An identity-init tanh MLP readout on even `(q, ξ, bump)`
  with epigraph L∞ then one refine later measured
  `max|r| = 7.847e-3` (`HΩ(0)≈+1.007`).
  Even compact Gaussians at the origin and `|y|≈1.50`
  with epigraph L∞ then one refine later measured
  `max|r| = 7.840e-3`.
  An identity-init SiLU MLP readout on even `(q, ξ, bump)`
  did not promote (LP pred earn `~1.4e-7`).
  Unfreezing the tanh hidden as additive `ΔW`/`Δb`
  did not promote (LP pred earn `~1.74e-6`).
  A multiplicative even Fourier on \(\xi=y^2/(1+y^2)\)
  did not promote (LP pred earn `~8.2e-5`).
  Identity-init GELU on \((\rho, E, \rho-q)\) did not
  promote (LP pred earn `~3.4e-7`).
  Identity-init paper MSNN on
  \(\eta=[(2/\pi)\arctan y]^2\) did not promote
  (LP pred earn `~1.87e-5`; far-field peak walk).
  Identity-init Mish on a log1p chart did not promote
  (tail-capped LP pred earn `~1.59e-6`).
  Even \(\operatorname{sinc}(y/s)\) realized
  `~7.0e-7` (below the `1e-6` promote gate).
  Even cylindrical \(J_0(y/s)\) did not promote
  (LP pred earn `~7.5e-7`; far-field peak walk).
  A residual-shaped even lift realized only `~3e-11`.
  Unfreezing the PirateNet last layer did not promote
  (LP pred earn `~7.4e-6`).
  Even Hermite–Gauss functions did not promote
  (LP pred earn `~3.6e-5`).
  Unfreezing stage-3 MSNN Fourier frequencies did not
  promote (LP pred earn `~3.4e-6`).
  Unfreezing the PirateNet skip-gate `α` did not
  promote (LP pred earn `~8e-9`).
  Identity-init tanh on an asinh chart did not
  promote (LP pred earn `~1.7e-6`).
  Even \(y\,\mathrm{dawsn}(y/s)\) did not promote
  (LP pred earn `~2.1e-7`; far-field peak walk).
  Unfreezing PirateNet embedding bias `Δbe` did not
  promote (LP pred earn `~2.6e-6`; Hilbert-quadratic
  ray `s=0`).
  Even \(\tanh(\sinh(y/s))^2\) did not promote
  (LP pred earn `~1.6e-7`; quadratic ray `s=0`).
  Even Whittaker \(e^{-z/2}U(1,1,z)\) did not
  promote (realized earn `~2e-11`).
  Paper L2 / grad-norm /
  exp-adjacent multiplicative raised 1601-pt L∞ on that
  family. A same-Jacobian bake-off ranked epigraph L∞ first;
  L2 GN / peak-weighted L2 / Lawson IRLS raised predicted L∞
  to `0.037` and earned 0. Random-embedding PI-init cannot
  represent that signed stack (hat error `0.31`, residual
  `0.10`, `HΩ(0)` negative) and is not a stand-alone champ path.

### Added — gauge-hard Hilbert-coupling homotopy GN

- `omnibias.jax.optim.homotopy_gauss_newton_minimize` runs Martens–Grosse
  or cubic GN on `r(θ, t)` at increasing coupling `t` (for residuals that
  split as `L + t Quad`). `peak_weighted_residual` is an L^∞ proxy that
  does not change the zero set. `champ_barrier_residual` is an
  active-set L^∞ residual plus a one-sided excess barrier so GN cannot
  raise already-small entries above a known champ.
  `linearized_linf_direction` is the Chebyshev (minimax) step on a
  linearized residual `r + J c` (exact on one column; Lawson IRLS on
  several). The campaign earn used the epigraph LP of that step.
- `omnibias.pinn.jax.discovery.ccf_hat_homotopy` lifts a *signed*
  Chebyshev hat through the official compactified envelope, hard-gauges
  `Ω(0.5)=0.05` inside the Jacobian, starts from a nodal hat (so
  `HΩ(0)` can be positive), runs frozen-velocity Picard, then optional
  coupling homotopy, and scores with `free_omega_vorticity_residual`.
  Prototype. An 11-node cubic Hermite plus frozen dual even
  Chebyshev-arctan, Gaussian / sech / compact-C¹ / rat4 / Laplace
  pads, then linearized L∞ Newton on `s=0.8`, `s=0.15`, `s=3.5`,
  `s=0.08`, `s=0.30`, `s=1.8`, `s=2.8`, tanh-chart Chebyshev
  `s=1.0`, Boyd-map Chebyshev `s=0.9`, tanh-sinh Chebyshev
  `s=1.0`, erf-chart Chebyshev `s=1.1`, p=4 algebraic Chebyshev
  `s=1.0`, asinh-arctan Chebyshev `s=0.85`, softsign Chebyshev
  `s=1.2`, even Legendre on a p=3 algebraic chart, even
  Chebyshev-U on a p=2 algebraic chart, even Gegenbauer
  \(C^{(3/2)}\) on a log1p-arctan chart, even Jacobi
  \(P^{(2,2)}\) on a p=6 algebraic chart, mapped Laguerre
  on \(\xi=y^2/(s^2+y^2)\), p=5 algebraic Chebyshev,
  half-stereo Chebyshev, even Gegenbauer \(C^{(5/2)}\) on
  a p=1.5 algebraic chart, associated Laguerre
  \(L_k^{(1)}\) on \(\xi=1-e^{-y^2/s^2}\) + compact C¹⁰ at
  `|y|=0.10` + multiplicative gauss–Welch at `|y|=0.10`,
  p=7 algebraic Chebyshev + compact C¹² at `|y|=1.60` +
  multiplicative Tukey \(\alpha=0.25\) at `|y|=1.60`,
  circular-chart Chebyshev \(x=2sy/(s^2+y^2)\) + Planck
  taper at `|y|=0.85` + multiplicative Planck at `|y|=0.85`,
  and erf-sinh Chebyshev \(x=\mathrm{erf}(\sinh(y/s))\) +
  flat-top at `|y|=0.10` + multiplicative flat-top at
  `|y|=0.10` family, then a signed PirateNet residual
  correction, then even Fourier stage-2, then identity-init
  even-`q` MSNN, then a second even-`q` MSNN (new seed),
  then even Chebyshev \(T_k(2q-1)\) on compactified `q`,
  then even Legendre \(P_k(2q-1)\) on compactified `q`,
  then even Padé `[4/4]` on \(x=2q-1\),
  then even Fourier \(k=6..12\) on compactified `q`,
  then an identity-init tanh MLP readout on even `(q, ξ, bump)`,
  then even compact Gaussians at the origin and `|y|≈1.50`,
  scored on 1601-pt L∞,
  later measured official-path `max|r| = 7.840e-3`
  (`HΩ(0)≈+1.007`; tanh-even parent was `7.847e-3`;
  Fourier-\(k=6..12\) parent was `7.876e-3`;
  Padé-on-`q` parent was `7.884e-3`;
  Legendre-on-`q` parent was `7.885e-3`;
  Chebyshev-on-`q` parent was `7.889e-3`;
  stage-3 MSNN parent was `7.938e-3`;
  first-MSNN parent was `7.943e-3`;
  even-Fourier parent was `7.991e-3`;
  PirateNet parent was `8.063e-3`;
  erf-sinh parent was `8.068e-3`;
  circular-Planck parent was `8.227e-3`; alg7 parent was
  `8.232e-3`; explag1 parent was
  `8.234e-3`; alg15-Gegenbauer parent
  was `8.278e-3`; half-stereo parent was
  `8.279e-3`; alg5 parent was
  `8.294e-3`; Laguerre parent was
  `8.297e-3`; alg6-Jacobi parent
  was `8.299e-3`; log1p-Gegenbauer parent
  was `8.309e-3`; alg2-U parent was
  `8.356e-3`; alg3-Legendre parent was `8.365e-3`; softsign parent was
  `8.37e-3`; asinh was `8.38e-3`; alg4 was `8.39e-3`; erf was
  `8.43e-3`; tanh-sinh was `8.44e-3`; Boyd was `8.46e-3`; tanh
  was `8.48e-3`; `s=2.8` was `8.50e-3`; `s=1.8` was `8.53e-3`;
  `s=0.30` was `8.56e-3`; L∞ parent was `1.008e-2`; Laplace was
  `1.24e-2`; rat4 was `1.25e-2`; C¹ was `1.28e-2`; peak-weighted
  `p=12` was `1.80e-2`; unweighted Hermite was `1.90e-2`; parent
  dual-scale plus Gaussian pads was `4.44e-2`).
  Stretch `1e-13` and `navier_stokes_proof_claim` stay unearned.
  Rung-1 is not started here.

### Added — JAX Wang residual + velocity integrate for free Ω

- `omnibias.pinn.jax.discovery.ccf_vorticity.wang_residual` is the JAX
  twin of the torch Wang residual. `free_omega_vorticity_residual` scores
  a free odd `Ω` with analytic `Ω_y`, `hilbert_wholeline_hp`, and
  `integrate_velocity_from_hilbert` (cumulative trap `U=∫_0 HΩ`). That
  `U` is not the `OperatorBlock` `integral` role. Stretch `1e-13` and
  `navier_stokes_proof_claim` stay unearned.

### Added — whole-line hp Hilbert for free neural Ω

- `omnibias.pinn.{jax,torch}.hilbert_line.hilbert_wholeline_hp` is a
  numerical split-core GL + power-mapped-tail Hilbert for a free (non-Hardy)
  odd `Ω`. Planted `H[Q]=-P` sits near `1e-14` (wide atom) / `1e-13`
  (narrow `a=0.25`); campaign GL96+raw-`u` tail sat at `1e-3`–`1e-1`.
  Torch `train_hilbert="wholeline_hp"` keeps the Wang net and swaps the
  operator. Not a certificate. Stretch `1e-13` and
  `navier_stokes_proof_claim` stay unearned until a trained residual
  measures it.

### Added — JAX cubic-regularised Gauss-Newton

- `omnibias.jax.optim.cubic_regularized_gauss_newton_minimize` is the
  matrix-free ARC twin of `omnibias.torch.optim.CubicRegularizedGaussNewton`.
  Discovery `GNConfig(method="cubic")` dispatches to it. Default
  `krylov_dim=None` uses the full parameter dimension (a fixed 20-D
  Lanczos cap silently stalled wide Phase-0 hops). `train_gn` enables
  JAX `float64`. Stretch `1e-13` and `navier_stokes_proof_claim` are
  unchanged.

### Added — CCF Hardy conjugate-tower CAP increment

- `hardy_conjugate_dictionary` / `spatial_odd_omega_atoms` plus closed-form
  Hardy-Ω velocity `U` (`U'=HΩ`, `U(0)=0`) in `omnibias.core.conjugate` and
  interval twins in `omnibias.core.verified.hardy_line`.
- Torch projection and JAX `hardy_omega_profile` accept `max_order` (default 0
  is today's Q-only path). Whole-line CAP
  `certified_ccf_hardy_wholeline_blowup_attempt` accepts `orders` / `parities`.
- `benchmarks/reproduce_deepmind_ccf.py --dictionary conjugate --max-order N`
  and `benchmarks/ccf_conjugate_sweep.py` measure residual vs span, Gram
  condition, and a CAP attempt. Stretch `1e-13` and Rung-1 `1e-11` are
  unchanged. `navier_stokes_proof_claim` stays false.
  `whole_line_certified` flips only on a genuine residual + dual NK close.

### Added — mapped-tail PV Hilbert (free neural Ω)

- `hilbert_pv_mapped_tail` / `train_hilbert="pv_mapped_tail"` adds a
  Gauss–Legendre `|t|>Y` tail and, when an `Ω` callback is available, a
  GL subtracted-kernel interior (planted core `~1e-3` at 96 nodes).
  Numerical, not a certificate. Stretch `1e-13` and Rung-1 stay unearned
  until measured.

### Added — public CSV equation discovery

- `examples/symbolic_discovery/public_csv_discovery`: Hudson Bay lynx–hare
  table (committed, offline) plus a synthetic Lotka–Volterra orbit.
  Train-only cubic-spline interpolant + STLSQ on `{1, x, y, xy}` (Huber
  on the public table, ridge on the clean orbit), scored by RK4 rollout
  against persist-last / linear / FD. Recovers `xy` signs on both splits;
  extra linear terms survive on the pelts. A spline-collocated RF jet
  takes the STLSQ seat when it clears 1.25× FD; otherwise the named
  cubic spline is the interpolant. Not a new law of nature.
- `omnibias.symbolic.fit_sparse_equation` accepts `loss="huber"` (IRLS,
  default remains ridge). `fit_neural_field_1d` accepts `weight_scale`
  (default `1.0`) and optional first-jet collocation (`deriv="spline"|"fd"`
  or `y_prime`; default value-only).
- `fit_field_jets_1d` fits a 1-D field on `fit_idx` only and refuses
  in-support claims outside that interval. `rollout_levels` /
  `rollout_skill` score an ODE by RK4 vs persist-last.
- `Interpolant1D` / `fit_cubic_spline_1d` is a named cubic-spline
  baseline (analytic `y'`), not a `NeuralField1D`. Spline-collocated
  jets can take the lynx–hare STLSQ seat when they clear 1.25× FD.

### Added — Path B operator completeness

- Wilson-rectangle transform `wilson_loops_to_ensemble_table` plus
  `static_potential_from_wilson`. New legal atoms: `V_r`, `t_wilson`,
  `creutz_chi`, `L_lat`, `F_r`, `sigma_lat`. `ensemble_table_from_mc_dict`
  now ingests `wilson_loops`.
- Lattice scale setting: `sommer_r0` (`r² F(r) = 1.65`) and SU(2) Luscher
  Wilson flow (`t0` / `w0` from `t² E(t)`). Not an `a→0` theorem.
- `PiecewiseEnsembleDiscoverer` (partition else-if on `T_lat`),
  `ensemble_field_law` (interpolant jet, not a jet of `A`),
  `ImplicitSystemDiscoverer`, and `CoupledConfinementDiscoverer`.
  `yang_mills_claim` stays false.

### Added — Path B confinement primitives

- `LatticeMetadata` on `EnsembleObservableTable` (`β`, `a`, shape, scheme,
  `already_inverse`, `n_configs`). New legal atoms: `area`, `perimeter`,
  `log_p2`, `inv_p2`, `ghost_G`, `log_ghost_G`, `T_lat`.
- `NamedFamilyDiscoverer` / `JointLawDiscoverer` in `omnibias.symbolic`
  (decoupling, Gribov–Stingl, area–perimeter, Lüscher; shared `σ`).
- `gluon_propagator_ensemble` and Landau ghost `ghost_propagator_p2` /
  `ghost_propagator_ensemble` (plane-wave Rayleigh; not Kugo–Ojima).
- `SU3LatticeLinkField`, Cabibbo–Marinari `run_lattice_mc(gauge_group="su(3)")`
  on 4⁴ smoke volumes, `finite_t_scan_table`.
- `extrapolate_in_a2` / `ContinuumFitResult`: fit-earned `continuum_claim`
  on a finite multi-spacing table. Sealed YM / transfer certificates stay
  `continuum_claim=False`. `yang_mills_claim` stays false.

### Added — tighter Haar and three-plaquette G1

- `su3_wilson_haar_coefficient` evaluates the CI irreps as one-argument
  functions of `Re χ_f` and intersects each cell with a centered form.
  `n_cells=8` still overlaps, so `finite_gauge_report` stays at 32.
  Not 4-D SU(3) Yang-Mills.
- `certified_hamiltonian_gap` passes the residual cover's labelled
  ground-state upper as Lehmann `λ1`. G1 is measured on the already
  certified three-plaquette Hamiltonian (`j_max=1`). The factor is
  recorded, not a claimed `5x`.

### Added — cellwise Haar and Wilson-character β-domain

- `su3_wilson_haar_coefficient` encloses each torus cell by interval
  range times cell area. The locked Lipschitz majorant stays only as a
  width comparison. `finite_gauge_report` now requires a certified
  SU(3) gap and locks `n_cells=32` (smallest of `{16, 32}` that
  separates). Not 4-D SU(3) Yang-Mills.
- `certified_wilson_character_beta_domain` records the infinite
  character-basis Wilson gap on `(1/4, 1/2, 1, 2, 4)`, including the
  polymer two-scale failure `1/4`. Kind `wilson_character_beta_domain`.
  Not a physical critical coupling.

### Added — finite gauge report and polymer β-domain

- `certified_polymer_beta_domain` records the polymer majorant's
  certified domain on a locked dyadic `β` grid (largest certifying
  point and the next grid failure). Kind `polymer_beta_domain`.
  Not a physical critical coupling and not `a → 0`.
- `finite_gauge_report` seals the existing engines on one named spec
  (polymer, Wilson character, Haar identities, a small SU(3) Haar
  transfer, two-plaquette gap with measured G1, strip RP, scaling
  table). Kind `finite_gauge_report`. Honesty stays hard-wired.
  The bundle is not a staircase to Clay existence.
- Smoke: `benchmarks/gauge_finite_report.py`.

### Added — YM-adjacent finite campaign 4

- Two-plaquette `j_max=2` certifies `λ1-λ0` at the locked coupling.
  Three-plaquette `j_max=2` is basis-only in CI.
- Longer strip `3×3` reflection positivity, and
  `su2_spatial_torus_transfer` on a `2×2` class-angle torus (finite
  3+1-D, not 4-D Yang-Mills). Kind `torus_reflection_positivity`.
- SU(3) Haar unlocks `max_dynkin=3` via Clebsch recurrence from locked
  `p, q ≤ 2` characters. Weyl dimension `(3,0)=10` is a finite Mathlib
  identity. Not a Bessel product.
- Polymer `counting="cluster"` keeps the first terms and a geometric
  tail at `BETA_LOCK`. Method tag `finite_polymer_cluster`. Still not
  Osterwalder-Seiler.
- `certified_gap_scaling_table` aliases the heat-kernel scaling report.
  Smoke: `benchmarks/gauge_gap_scaling.py`. `continuum_claim` stays
  false. The continuum limit is not taken.

### Added — YM-adjacent finite campaign 3

- Default two-plaquette magnetics are Racah 6j recoupling weights
  (`magnetic="sixj"`); `magnetic="character"` keeps amplitude 1.
  `su2_three_plaquette_hamiltonian` certifies `λ1-λ0` on the 3-square
  chain at `j_max=1`. Kinds stay separate
  (`two_plaquette_hamiltonian_gap`, `three_plaquette_hamiltonian_gap`).
- Strong-coupling default is the two-scale remainder
  `u + A u²/(1-B u)` with first-step `A=20` and subsequent `B=15` in
  4-D. `BETA_LOCK` is `1/5`. Method tag `two_scale_polymer_count`.
  Single-scale `C=15` is not sold as `N_2`. Still not Osterwalder-Seiler.
- SU(3) Wilson Haar uses midpoint plus a Lipschitz remainder and unlocks
  `max_dynkin=2` with locked trigonometric characters. Not a Bessel
  product and not 4-D SU(3) Yang-Mills.
- `su2_spatial_strip_transfer` is a finite 2+1-D class-angle strip.
  `certified_strip_reflection_positivity` and
  `certified_strip_cluster_tail` are statements about that matrix.
  Smoke: `benchmarks/gauge_spatial_strip.py`.
- `OmnibiasAnalytic.Check.SixJ`, `Check.HaarVolume`, and Polymer
  `first_step_4` (`20`, `15<20`) are finite rational identities. The
  Mathlib bridge registers them before `sign`. Not a continuum claim.

### Added — two-plaquette SU(2) Hamiltonian gap

- `su2_two_plaquette_hamiltonian` and `certified_hamiltonian_gap` certify
  `λ1-λ0` of one finite Kogut–Susskind operator on Gauss-law triples
  `|j1,j2,js⟩`. Holonomy trials are measured against the standard basis
  (G1 factor is recorded, not invented). Sealed
  `verified-hamiltonian-gap-1` registers as `two_plaquette_hamiltonian_gap`.
  Not a continuum or Yang-Mills claim.

### Added — SU(3) Wilson transfer via enclosed Haar

- `su3_wilson_transfer` builds a diagonal character-basis matrix whose
  eigenvalues are interval enclosures of the Wilson Haar integrals on the
  SU(3) torus (`(p,q)≤1` truncation). Not a product of ordinary `I_n`,
  not 4-D SU(3) Yang-Mills.

### Added — backtrack polymer majorant and named coordination identities

- `certified_strong_coupling_glueball_bound` defaults to the
  backtrack-excluding tree count `C = 3(2d-3)` (`15` in 4-D) and locks
  `BETA_LOCK = 1/4`. The older `C = 24` overcount stays available as
  `counting="crude"`. Method tag `backtrack_polymer_count`; still not
  Osterwalder-Seiler and not a continuum claim.
- `OmnibiasAnalytic.Check.Polymer` proves `3*(2*4-3)=15` and `15<24`.
  `polymer_certificate` seals a locked family (`polymer` obligation class).

### Added — holonomy trial spaces on finite transfer matrices

- `holonomy_trial_space` builds gauge-invariant character trials on the
  dense `angle` / class-angle grid. `certified_transfer_matrix_gap(...,
  trial=)` adds a Lehmann–Maehly candidate; existing callers are unchanged.
  Gram condition numbers are sealed and flagged above a stated threshold.
  Certificates stay `continuum_claim=False` / `yang_mills_claim=False`.
- `benchmarks/gauge_holonomy_gap.py` records the measured G1 tightness
  factor (not a claimed `5x`) for one fixed matrix at one spacing.

### Added — crude strong-coupling polymer bound (fixed β)

- `certified_strong_coupling_glueball_bound` interval-certifies
  `m a ≥ -ln(C u(β))` for SU(2) Wilson at one coupling, with locked
  coordination `C = 8(d-1)` and `u = I₂/I₁`. `certified=True` only when
  `C u < 1`. Method tag `crude_polymer_count`; not a continuum claim.
- `certified_wilson_character_gap` encloses `-ln(I₂/I₁)` for the infinite
  character-basis Wilson transfer (0+1-D). Sealed
  `verified-strong-coupling-gap-1` certificates register on the gauge
  proof machine as `strong_coupling_glueball_gap`.

### Added — named SU(2) / SU(3) Casimir identities in Lean

- `OmnibiasAnalytic.Check.Casimir` proves the locked Freudenthal evaluations
  `C2(1)-C2(0)=3/4` (SU(2)) and `C2(1,0)=4/3` (SU(3)). Finite rational
  identities, not a continuum gauge claim.
- `casimir_certificate` seals a locked family. The Mathlib bridge
  re-derives the rationals and applies the Check theorem (`casimir`
  obligation class). Existing leaf classes stay leaf-only.

### Added — compact-box residual and finite-matrix gap in Lean

- `OmnibiasAnalytic.Check.Compact` proves a named incompressible residual
  lower bound on `[1/2, 1]²` and the characteristic-polynomial gap of
  `[[13/2, 3/2], [3/2, 13/2]]` (ratio `5/8`). Not a continuum regularity
  theorem or a continuum gauge claim.
- `compact_box_certificate` seals a locked family. The Mathlib bridge
  re-derives the rationals and applies the Check theorem (`compact_box`
  obligation class). Existing leaf classes stay leaf-only.

### Added — named NK unique-zero instances in Lean

- `OmnibiasAnalytic.Check.Kantorovich.Named` proves unique roots of three
  named polynomials on explicit compact boxes: circle ∩ line on
  `[5/8, 7/8]²`, Hopf radial `r(1-r²)` on `[3/4, 5/4]`, and Chebyshev
  `T₃` on `[3/4, 1]`. Not a continuum PDE, Lohner return map, or
  continuum CCF blow-up.
- `named_zero_certificate` seals a locked family. The Mathlib bridge
  re-derives the rational box and applies the Check theorem
  (`named_zero` obligation class). Existing leaf classes stay leaf-only.

### Added — enclosure-trace replay in Lean

- `OmnibiasAnalytic.Check.Enclosure` replays a finite rational
  `+ − × abs recip` DAG (`QInterval` / `evalTrace`). Locked plants cover
  tower Horner, NK bounds plus the unique root of `x² - 2`, `B₂` / named
  `zetaNeg1`, and a 2×2 LDLᵀ. Not a transcendental enclosure or analytic
  continuation.
- `enclosure_trace_certificate` seals a locked family. The Mathlib bridge
  re-derives the DAG and applies the Check plant theorem
  (`enclosure_trace` obligation class). Existing leaf classes stay leaf-only.

### Added — NK / Krawczyk existence in Lean

- `OmnibiasAnalytic.Check` proves 1-D Banach / Newton-operator unique-root
  theorems on a compact interval, plus a planted unique root of `x² - 2` in
  `[5/4, 7/4]` (radii and Krawczyk routes). Not a continuum PDE claim.
- `nk_existence_certificate` seals the locked plant. The Mathlib bridge
  re-derives the rational contraction facts and applies the Check lemma
  (`nk_existence` obligation class). Existing `radii_polynomial` / `krawczyk`
  classes still certify only `p(r) < 0` / containment.

### Added — Riccati / Eulerian tower in Lean

- `OmnibiasAnalytic.Tower` proves the sigmoid / tanh / sech / Hermite
  recurrences on `ℤ[X]`, the first-derivative Riccati identities, and the
  `iteratedDeriv` link theorems (softplus is the shifted sigmoid tower).
  Pointwise `C^∞` / algebraic identities; not a finite-difference collapse.
- `tower_coeffs_certificate` seals exact integer coefficients from
  `omnibias.core.verified.coeffs`. The Mathlib bridge re-derives the list and
  checks it against the Lean recurrence (`tower_coeffs` obligation class).

### Added — gauge ensemble language (Path B)

- `EnsembleObservableTable` / `LEGAL_ENSEMBLE_ATOMS` are the third language:
  ensemble statistics (`|P|`, `χ_P`, `C_P(r)`, Landau `G(p^2)`, planted `ρ`),
  not a jet and not a per-config loop table. One `LatticeLinkField` raises.
- `landau_gauge_fix` / `gluon_propagator_p2` evaluate a lattice Landau 2-point.
  `reconstruct_spectral_density` is a Tikhonov inverse of a finite
  Källén–Lehmann kernel; planted-`ρ` recovery is the only gate.
  Unregularized inversion and mass-gap claims raise.
- `StatisticalLawDiscoverer` recovers planted scaling / Polyakov-mass /
  area-perimeter / spectral laws. GEVP / transfer-gap stay certificates.
  Not a continuum or Yang-Mills mass-gap claim.

### Added — gauge loop language (language trap)

- `LEGAL_LOOP_ATOMS` / `LoopObservableTable` / `evaluate_loop_atoms` evaluate
  plaquette, `W(R,T)`, and Polyakov on `LatticeLinkField`. Creutz is a
  derived identity. `LoopLawDiscoverer` recovers `W(1,1) = plaquette` and a
  planted area law. Mixing a jet with loop atoms (or a Green function into
  a jet) raises. Not a continuum mass-gap claim.

### Added — gauge data paths (noise amplification)

- `ConnectionSource` / `LatticeLinkField` name the legal sources. Path D
  (lattice or mesh links interpolated by a random-feature field, then
  jetted) raises, as do `GaugeCovariantJet.from_lattice_links` /
  `from_neural_fields` and 1-D Fredholm/Volterra-as-YM-weak-form.
- `weak_yang_mills_residuals` / `evaluate_weak_ym_identity` contract
  `D*F` against a Gaussian adjoint test 1-form bank. Identity check, not
  a continuum or mass-gap claim. TV / GP are not product features.

### Added — gauge-invariant dictionary (search-space trap)

- `GaugeInvariantDictionary` / `enumerate_gauge_invariants` generate
  mass-dimension-graded Weyl singlets of `F` and `D F` before STLSQ.
  Named SU(3) census: coordinate 2-jet is 480; searchable dim-4 singlets
  are 2. Bianchi is an identity; `|F-*F|^2` is a Euclidean syzygy.
  Flattened `D^k F` component libraries raise. Complexity is
  representation-theoretic. Not a Hilbert-series completeness claim.

### Added — gauge-covariant jet wrapper (coordinate trap)

- `omnibias.geometry.gauge._core.covariant_jet.GaugeCovariantJet` stores
  `F` and `D_rho F` only. `LEGAL_SINGLET_ATOMS` is a closed allowlist;
  `assert_library_gauge_legal` and `evaluate_gauge_law_gate` are fail-closed.
  New kernel `covariant_derivative_field_strength` (torch / jax twins).
  Public `bpst_instanton_arrays`.
- Optional `omnibias.symbolic.gauge_discovery.GaugeLawDiscoverer` (extra
  `omnibias-symbolic[gauge]`) recovers classical local singlet identities
  (BPST `tr(F^2) ~ 8 pi^2 tr(F*Ftilde)`). Not a `FieldJet` of `A`, not
  Wilson / Polyakov language, not a continuum mass-gap claim.

### Added — remaining Group 02 gated architectures (01-03, 01-06, 01-08–01-10, 01-12, 02-02, 02-06–02-14)

- Arrangement geometry in `omnibias.partition.arrangement`. Cell membership
  is temperature collapse (`beta -> inf`), not founding `delta -> 0`.
  Sampling is a subgraph. Sound gap, not P vs NP. No new package.
- OMBU frames in `omnibias.core.frames`. `sigma'` is not admissible;
  frames are not orthonormal and not compactly supported. Pack order
  remains a band selector (01-07), not a Littlewood-Paley claim.
- Tropical homotopy in `omnibias.struct._core.tropical`, reusing
  `logsumexp_gap_bound`. Large `(n, D)` refused. G4 path-following is
  `--full` only.
- Equality locus in `omnibias.core.locus` + `omnibias.fields.locus`.
  Constraint manifold, not a general PDE solver; always returns
  `branch` / `condition` / `converged`.
- Jet-bundle vocabulary in `docs/theory-jets.md` + `contact_residual` /
  `is_holonomic`. Dictionary, not a discovery; no `omnibias-jetbundle`.
- Conjugate Hilbert tower extending `hardy_line` plus
  `omnibias.core.conjugate` / `{torch,jax}.conjugate`. Line Hilbert only;
  G5 is a projection defect, not a stretch-gate clearing, and is not in
  CI `all_passed`.
- Face-Net in `omnibias.graph.arrangement` on a sampled tope subgraph.
- BEM-Net in `omnibias.pinn.bem`: PDE exact off-surface; BC approximated;
  linear constant-coeff homogeneous only.
- Hierarchical pack tree in `omnibias.core.hierarchy` +
  `{torch,jax}.hierarchy`. 1-D offsets; `eta=0` bit-identical to dense.
  G3 complexity smoke-recorded.
- Equivariant / chart scan: gaussian-family steering only; discrete
  `C_L`, not SO(2)/SO(3).
- Tanh-method solitons: tanh algebra, not a collapse; multi-kink is not
  the n-soliton formula.
- Hermite ladder: raw tower is not the QHO eigenbasis; Rodrigues
  reweight required. Anharmonic G5 may lose.
- Layered transfer: 1-D ABCD stacks; `continuum_claim=False`; distinct
  from `geometry.gauge.transfer`.
- Named linearizing transforms (Cole-Hopf / Miura / Bäcklund / Darboux).
  Exactness to jet order N. 03-11 search stays designed.
- Holonomy band wrapping existing gauge transport. Closed form only
  abelian + transverse-constant; open lines gauge-dependent; no
  Yang-Mills / mass gap / continuum claim.

Cost / wall-time / FermiNet-many-body gates are smoke-earned, not in CI
`all_passed`. Zero new packages.

### Added — Wave-3 gated algebra + architectures (01-05, 01-07, 02-01, 02-03, 02-04, 02-05)

- Mollifier calculus in `omnibias.core.mollifier` (`MollifierSpec`,
  `tail_bound`). Analytic bases have certified exponential tails, not
  compact support; higher-order kernels take negative values. Gated
  (G1–G3 earned; G4 deferred to VPINN).
- Spectral design in `omnibias.core.spectral_design` (`BandPlan`,
  `peak_frequency`). Pack order is a band selector, not a Littlewood-Paley
  completeness claim. Gated (G1–G2 CI-gated; G3 not in CI `all_passed`).
- Scan-Net torch `nn.Module` + jax pytree twins. Equivariance is
  per-layer, per-direction, on-lattice, not translation on `R^D`; `gamma`
  is not `delta -> 0`. Templates reuse the six `OperatorBlock` roles.
  Gated (G1/G2/G5 CI-gated; G3 cost and G4 k-NN recorded, not in CI
  `all_passed`).
- Jet-KAN twins. Exactness is of the model jet, not the target; the
  Kolmogorov-Arnold theorem does not justify the architecture. Gated
  (G1/G3/G5 CI-gated; G2 jet-vs-autodiff timing smoke-earned, not in CI
  `all_passed`).
- Weak-form VPINN in `omnibias.fields.weak`. Exact integrals only for
  polynomial coefficients on boxes; path recorded; boundary bound on by
  default. Gated.
- Multi-interface transmission PINN in `omnibias.pinn.interface` (alias
  `TransmissionInterface`). `alpha -> inf` is interface sharpening,
  neither collapse; parallel interfaces only. Distinct from XPINN
  `omnibias.pinn._core.interface`. Gated.

### Added — Wave-1 primitives (01-04 irregular stencils, 01-02 bias scan)

- Exact rational irregular / Birkhoff finite-difference weights in
  `omnibias.difference` (`solve_irregular_stencil` over ``Q``,
  `is_poised_exact`, `certified_irregular_error`). Scale-free ``A_{i,p}``
  satisfy ``a = A / h^{q-p}``; order is asymptotic in the node scale ``h``.
  Gated (G1–G4 earned).
- Transverse bias scan: `BankSpec` plus torch `BiasScan` and a jax functional
  twin. Equivariance is an interior lattice shift along ``w``, not a circular
  wrap; soft-argmax ``gamma`` is not ``delta -> 0``. Gated (G1–G3 CI-gated;
  G4 earned on smoke vs voxelized ``cmbConv1d``, not in CI ``all_passed``;
  two-interface soft-argmax bias recorded, not a win).

### Added — Equinox CI, unplanted Arrangement, Keras Boosted beta

- Equinox wrapper tests **fail** when ``CI`` is set and the extra is missing
  (local runs still ``importorskip``). CUDA AMP stays skip-if-no-CUDA.
- Arrangement tab-head on the switched ODE is unplanted (constructor random
  ``W``, no ``e_0`` axis init); fitted ``W``/``t`` still recover both field-jet
  laws.
- Keras ``ArrangementBoosted(learnable_beta=True)`` tape: member ``beta`` is
  trainable and has nonzero grad through the ensemble forward. The layer
  implements ``build()`` so Keras 3 does not warn on first call.
- Cookbook recipes ``tab-as-layer`` and ``piecewise-hybrid-automaton``; plugin /
  piecewise honesty in API, READMEs, and consumer skills.

### Added — four remaining test / coverage gaps

- Tab-head switched-ODE path trains SoftTree **and** Arrangement (``H=1``
  regression) on the trajectory's finite-difference ``du`` (kinked; the
  field-jet ``du`` is smoothed), then hardens the **fitted** ``W``/``t``
  (no ``_refine_split_threshold`` polish). STLSQ still uses the field jet.
  ``fit_learned_piecewise_ode`` still polishes its own learned gates.
- Per-leaf ``NeuralJetDiscoverer`` requires both regimes on the **same**
  two-sided tree; identity-passing far leaves with enough jet variation
  must recover ``c_y`` (compact leaves stay unidentified).
- ``torch.compile`` on all three heads is required when ``CI`` is set (CPU
  inductor); local runs still skip if inductor is missing. CUDA AMP stays
  skip-if-no-CUDA.

### Added — seven remaining honesty / coverage nits

- Tab-head switched-ODE path (superseded below): a depth-1 SoftTree can be
  hardened via ``tree_params`` and fed to ``fit_piecewise_law`` on a field jet.
  ``fit_learned_piecewise_ode`` stays the differentiable-partition control.
- ``certify_composed``: SoftTree / Arrangement encoders (optional Linear IBP
  prefix) get a sound interval latent box (``tab+tab`` / ``ibp+tab+tab``);
  arbitrary modules stay ``sampled_latent`` (not a sound enclosure of ``E(box)``).
- Tab CI smokes ``KERAS_BACKEND=torch`` on ``test_keras.py`` (torch already in
  the tab job; TensorFlow stays a dedicated step, not a required dep). Keras
  ``ops.prod`` on the torch backend dropped float64; ``prod_last_axis`` keeps dtype.
- Depth-1 ``extract_tree_jet_directional`` uses ``mlp_jet``; depth ``>= 2`` keeps
  the Leibniz product.
- Equinox ``BoostedHead`` wrapping ``boosted_forward``; ``import omnibias.tab.jax``
  stays Equinox-free.
- Learned vector hybrid (``fit_learned_piecewise_ode`` on two-component ``u``)
  without an oracle partition; oracle vector test stays the control.
- Per-leaf NeuralJetDiscoverer scans every tree's ``hard_assignment`` after an
  unplanted fit (not planted tree 0).

### Added — eight remaining honesty / coverage gaps

- `NeuralJetDiscoverer.include_x` (default ``True``); per-leaf two-regime gate
  uses the sparse equation's ``c_y``, not an ``lstsq`` of the jet.
- Learned trajectory split is fit on a closed-form field jet
  (``fit_neural_field_nd`` + ``extract_field_jet``); algebraic i.i.d. samples
  and the example's oracle partition stay controls.
- ``as_head`` returns ``TabHead`` (logits ``(..., k)``; attributes forward to
  the inner module).
- Keras ``ArrangementBoosted`` (stacked ``keras.ops`` combine),
  ``learnable_beta`` on SoftTree / Arrangement, and a TensorFlow smoke of
  ``test_keras.py`` (TensorFlow is not a required dep).
- ``certify_composed``: IBP when Linear/activation ingest works (including a
  flattenable ``ModuleList``); otherwise ``method="sampled_latent"`` (not a
  sound enclosure of ``E(box)``).
- Depth-1 ``extract_tree_jet`` uses ``mlp_jet_mv`` (additive Linear-sigmoid-
  Linear); depth ``>= 2`` keeps the Leibniz product.
- ``fit_first_order(..., encoder=)`` joint-Adam test; always-on CPU AMP
  (bfloat16) over all three heads; ``torch.compile`` skips only if inductor
  is missing.

### Added — remaining plugin / discovery gaps

- AMP and `torch.compile` parametrized over SoftTree / Arrangement / Boosted
  (CPU AMP always-on; CUDA AMP skip if no CUDA; compile skip if no inductor).
- `learnable_beta` tests: `_beta` is an `nn.Parameter` with nonzero grad and
  follows `.to(dtype)` on SoftTree and Arrangement.
- Per-leaf NeuralJetDiscoverer gate on a two-regime `exp` / `exp(-2x)` **soft-tree
  jet** (`delta -> 0`); `extract_tree_jet` vs autodiff. Switches remain
  `beta -> inf`.
- Learned partition on the switched-ODE **trajectory**
  (`fit_learned_piecewise_ode`); algebraic i.i.d. samples stay the control.
- Optional Equinox extra: `omnibias.tab.jax.equinox_head` (`eqx.Module`);
  `import omnibias.tab.jax` stays Equinox-free. CI `pip install equinox` next
  to Keras.
- Optional keyword-only `encoder=` on `fit_first_order` / `fit_second_order` /
  `fit_arrangement` (default `None` is the tabular G3 path). Stagewise
  `fit_boosted` / `fit_arrangement_boosted` raise `TypeError` pointing at
  `fit_joint` / `fit_second_order`. G3 JSON untouched.

### Added — tab plugin completion + if-else neural-jet discovery

- `as_head(z, kind)` returns a `TabHead` wrapping SoftTree / Arrangement /
  Boosted on `z.device` / `z.dtype`; arrangement logits are `(..., k)` with numpy
  `predict` squeezing `k=1`; `beta` / `lr` are buffers; batched
  `ArrangementBoosted.forward`; `fit_joint(encoder, head, X, y)` without
  changing G3 `fit_*` signatures; multiclass / regression `cell_logits`.
- Keras 3 layers in `omnibias.tab.keras` and `omnibias.partition.keras`
  (`keras.ops`, not `omnibias-keras`), including `ArrangementBoosted` and
  `learnable_beta`.
- `certify_composed(encoder, head, box)`: IBP the encoder when ingest works,
  else `sampled_latent` (not a sound enclosure of `E(box)`); grid+random
  soundness on composed samples.
- Soft-tree jets: `extract_tree_jet` / `extract_arrangement_jet`. Depth-1 uses
  `mlp_jet_mv`; depth `>= 2` is the Leibniz product of sigmoid jets (exact for
  the **soft** surrogate).
- Learned-partition piecewise discovery: `fit_learned_piecewise_ode` trains
  differentiable soft-weighted per-cell models then STLSQ-polishes; vector
  `HybridAutomaton` with shared gates. Oracle partitions stay the control.

### Added — tab layers as neural-net heads (plugin contract)

- `ArrangementBoosted` is an `nn.Module`: `forward` is `base + lr * sum members`
  with autograd through every weak learner (no `detach` / numpy).
- `ArrangementClassifier`, `ArrangementBoosted`, and `SoftTreeEnsemble`
  `forward` accept leading dims `(..., d)` and compose with any encoder;
  constructors stay float64 CPU (certify / G3). Plugin use is
  `.to(device=z.device, dtype=z.dtype)` then the user's optimizer.
- JAX twins: `omnibias.tab.jax.arrangement_forward` /
  `boosted_forward` (float64 parity `~1e-9`).
- Gate: `packages/omnibias-tab/tests/test_plugin.py` plus
  `docs/examples/tab_as_layer.py` and `tab_as_layer_jax.py` (CI).

### Changed — fair early-stop protocol for arrangement vs LightGBM

- `fit_arrangement` now early-stops on validation BCE with best-checkpoint
  restore (`patience` / `eval_every` / optional outer `X_val`); `steps` is a
  max cap. Both G1/G2 and G3 runners train on `Xtr` only for arrangement and
  LightGBM (no train+val refit); `--full` rejects arrangement arms that hit
  the Adam step cap without plateauing.
- Regenerated smoke + full artifacts under the fair protocol (previous G3
  win/loss numbers are not comparable).

### Added — 05-02 G3b capacity suite (G3 frozen)

- `fit_arrangement(..., optimizer="trust_region"|"cubic")` and
  `fit_arrangement_boosted` (Newton-boosted H=2 arrangements, stage-wise
  val-BCE early stop).
- `benchmarks/tabular_arrangement_capacity.py`: parallel arms (`h2_adam`,
  `h2_newton`, `h3_adam`, `h4_adam`, primary `boost_h2`, `tab_boost`,
  `tab_joint`) vs the same fair LightGBM. G3b earned only if predeclared
  `boost_h2` is not-worse on >=6/8 public sets. Smoke:
  `docs/benchmarks/tabular_arrangement_capacity_smoke.json` (CI); `--full`
  writes `docs/benchmarks/tabular_arrangement_capacity.json`
  (`g3b_earned: false`; primary not-worse on 4/8).

### Added — 05-02 G3: public arrangement vs LightGBM win/loss table

- `omnibias.tab.bench.ARRANGEMENT_PUBLIC_SUITE`: eight binary public datasets
  (breast_cancer offline; adult / higgs / banknote / blood_transfusion /
  ionosphere / sonar / spambase via OpenML with skip-on-fail); adult encoding
  fixed; `train_val_test_split` for stratified 60/20/20.
- `benchmarks/tabular_arrangement_public.py`: tuned LightGBM + H=2
  `fit_arrangement` (dense + sparse arms); per-dataset win/loss table; G4
  diagnostic predictiveness reported with a frozen threshold (not retuned).
- Artifacts: `docs/benchmarks/tabular_arrangement_public_smoke.json` (CI) and
  `docs/benchmarks/tabular_arrangement_public.json` (`--full`).

### Added — Wave-1 primitive 01-01: multipack Birkhoff collapse

- `omnibias.core.multipack`: `PackSpec` / `MultiPackSpec`, Polya screen,
  incidence matrix, `is_poised` (returns `None` when inconclusive).
- `omnibias.torch.multipack.MultiPackUnit` (+ `BirkhoffOMBU`) and the
  bit-identical `omnibias.jax.multipack` twin: closed-form
  `sum_g c_g sigma^(n_g)(z + mu_g)` with shared means.
- `benchmarks/multipack_birkhoff.py`: G1/G2/G3/G5 earned; float64 order
  ceiling recorded (sigmoid: 1, tanh: 2 on `z in [-1,1]`); G4 deferred.
  Smoke: `docs/benchmarks/multipack_birkhoff_smoke.json`.

### Changed — A4 hardening (05-02 protocol repair)

- Removed process-global `torch.set_default_dtype` from
  `omnibias.tab.torch.arrangement` (module constant `_DTYPE` instead).
- LightGBM baseline: extended grid + early stopping; `--full` refuses a
  selected config on the grid boundary before reading G1/G2.
- `obliqueness_diagnostic` honesty: records that it does not discriminate
  XOR from axis (linear oblique only); not retuned on G1/G2.
- Wired `certify_arrangement_gap` (partition soft->hard gap reuse); tie-aware
  AUC; vectorized sparse warm-starts; arrangement budget recorded per tier
  (smoke = wiring gate).

### Added — Wave-0 falsifier A4: tabular arrangement vs LightGBM (05-02 G1/G2)

- `omnibias.tab.arrangement` / `omnibias.tab.torch.arrangement`: H-hyperplane
  soft arrangement classifier on `omnibias.partition` POU weights, with beta
  anneal, L1 on normals, and a sparse feature-pair warm-start.
- `benchmarks/tabular_arrangement.py`: constructed oblique XOR (G1) and axis
  AND (G2) vs a val-selected LightGBM grid over five seeds; worst-seed gates
  via `require_all_seeds`. G1/G2 earned; G3–G7 unearned. Spec 05-02 status
  `concept -> gated`.
- Artifacts: `docs/benchmarks/tabular_arrangement_smoke.json` (CI) and
  `docs/benchmarks/tabular_arrangement.json` (`--full`).

### Changed — A7 hardening (05-01 G7 protocol repair)

- `benchmarks/inverse_imaging.py`: five-seed worst-case gate via
  `require_all_seeds`; vectorized `polish_peak_batch`; derived
  `boundary_contamination_ratio` guard and `s = round_down_1sig(0.5 * s_max)`;
  discloses the locally-seeded coarse grid and reports a `tau*`-free global
  argmax arm (`global_search_earned`: n=3 true, n=4 false, boundary artifact
  recorded). Both tiers use `R = 128` per seed.
- Gate helper `require_all_seeds` in `benchmarks/_gates.py` (worst-seed gate;
  refuses `n_seeds < 5` without an explicit override).

### Added — Wave-0 falsifier A7: scan localization scaling (05-01 G7)

- `benchmarks/inverse_imaging.py`: logistic bias-scan localizer; gates
  `sd(tau_hat) ~ alpha^(n - 5/2)` for `n in {3, 4}` (fitted exponents within
  `0.1` over 1.2 decades), rational `||sigma'''||_2^2 = 1/42` /
  `||sigma''''||_2^2 = 1/30`, and discrete `sd(r')` prediction. G1–G6 remain
  unearned; no `omnibias.pinn.inverse` module.
- Gate helper `require_capture_rate` in `benchmarks/_gates.py` (raises
  `RuntimeError` / `INVALID EXPERIMENT` on a broken regime).
- Artifacts: `docs/benchmarks/inverse_imaging_smoke.json` (CI) and
  `docs/benchmarks/inverse_imaging.json` (`--full`); G7 earned. Spec 05-01
  status `designed -> gated`.

### Added — Wave-0 falsifier A6: Fisher pack degeneracy (04-01 G2)

- `benchmarks/information_geometry.py`: closed-form integrand + Monte Carlo
  Fisher for the two-bias logistic pack; gates exponent `2.00 +- 0.02` over
  three decades of pack spread `delta` and prefactor `1/720`.
- Gate helpers `require_scaling_exponent`, `require_rel_error`,
  `require_within_stderr` in `benchmarks/_gates.py`, self-tested in
  `tests/test_gates_protocol.py`.
- Artifacts: `docs/benchmarks/information_geometry_smoke.json` (CI) and
  `docs/benchmarks/information_geometry.json` (`--full`); G2 earned, G1/G3–G5
  remain unearned. Spec 04-01 status `concept -> gated`.

### Fixed — Phase-0 CCF reproduce audit (P0 honesty / warm / anti-ghost)

- Warm-start: architecture mismatch rebuilds a fresh net (`strict=True`); no
  hybrid `strict=False` leftover weights.
- Anti-ghost uses **pre-rescale** gauge / peak samples so hard gauge rescale
  cannot hide ghosts.
- CLI / `run_once` default train Hilbert is `hardy_corrected_pv` (matches
  `reproduce_deepmind_config`); multistage GN labeled `gauss_newton_corr_proxy`
  (corr-matching, not Wang-linearized).
- Escalate warm lineage syncs `reproduce` ↔ `ab`; multistage re-score uses
  `max(dense, projection_defect)`.

### Added — Phase-0 DeepMind neural CCF reproduction (stretch 1e-13)

- `reproduce_deepmind_config` / Martens–Grosse primary on compactified neural Ω
  with **`hardy_corrected_pv`** train Hilbert; dense Wang metric on the neural
  profile (`dense_neural_vorticity_residual`) with anti-ghost.
- `benchmarks/reproduce_deepmind_ccf.py` smoke + escalate loop; campaign tick
  starts at `phase0_reproduce_neural` until stretch clears (then Hardy Rung-1/2).
- Multistage `iterate_multistage` with labeled `stage2_heuristic_adam` and
  optional `gauss_newton_corr_proxy`.
- Stretch \(10^{-13}\) and Rung-1 \(10^{-11}\) **unchanged / unearned**; measured
  floor ~\(10^{-1}\) (Hilbert × dictionary catch-22 — not cleared by more MG).

### Added — Exact Martens–Grosse Gauss–Newton (QR / CGLS, exact JVP)

- `omnibias.jax.optim`: `lstsq_gauss_newton_direction`, `cgls`,
  `gauss_newton_direction_cgls`, `martens_grosse_combine` (exact JVP — no FD
  probes), and `martens_grosse_gauss_newton_minimize` (default `solver="qr"`,
  `use_martens_grosse=True`).
- Torch twins: `martens_grosse_combine`, `martens_grosse_gauss_newton_minimize`,
  and `GaussNewton(..., use_martens_grosse=True)`.
- Discovery `train_gn` is a thin pytree wrapper over the JAX primitive; CCF
  Hardy earn path records `train_hilbert=hardy_exact_omega`, `gn_solver=qr`.
  Absolute Rung-1 residual gate unchanged / still unearned.

### Added — DeepMind campaign autonomy (Hardy-aligned earn path, adapters, Phase 5)

- Earn-path default train Hilbert is now **Hardy projection** (matches Rung/CAP);
  truncated-line spectral `H` remains diagnostic-only. Adam forbidden on
  `CCFHardyAdapter` earn path (Martens–Grosse Hardy-Ω GN).
- Acceptance ladder prefers JAX Martens–Grosse vs torch CubicGN by dense residual;
  multistage uses Hardy Wang residual.
- `benchmarks/deepmind_campaign_tick.py` for Cursor `/loop`; skill
  `omnibias-dev-deepmind-campaign` + rule `deepmind-campaign.mdc`.
- Pipeline adapters: `CCFHardyAdapter` (vorticity), `IPMAdapter`, `BoussinesqAdapter`.
- Phase 5 beyond-DeepMind helpers in `phase5_beyond.py` (partition / tab / logic
  gates; blocked until Rung-2). Scaffold IPM/Boussinesq smoke + `_gates.py`
  helpers. Absolute CCF residual gates **unchanged** (`1e-11`); still unearned
  until dense Wang clears under anti-ghost.

### Added — CCF five-point residual push (grad-norm, d1/d2, linearized MSNN, hybrid H)

Torch vorticity discoverer now trains with gradient-normalized Wang residual,
d1/d2 residual stack + hybrid adaptive collocation, softplus multiscale core,
CubicGaussNewton early-stop, and (historically) hybrid Hilbert. **Superseded for
earn path** by Hardy-aligned train Hilbert above. Acceptance `--full` budgets
default to serious CubicGN/QR/MS counts under `$OMNIBIAS_SCRATCH`, wrapping
`$OMNIBIAS_SUBMIT` when set. Absolute gates \(10^{-11}\) / \(10^{-6}\) / stretch
\(10^{-13}\) remain **uncleared** until measured.

### Added — CCF DeepMind residual push (neural CubicGaussNewton, vorticity end-to-end)

Torch compactified neural vorticity discoverer
(`omnibias.pinn.torch.discovery.ccf_vorticity_neural`) with OMBU closed-form
\(\Omega_y\), Hardy-projection exact Hilbert (not truncated FFT as the Rung
metric), CubicGaussNewton primary + QR polish, multistage correction, optional
funnel \(\lambda\), and vorticity-form mpmath polish / CAP. Far-field Ω-primary
Hardy dictionary cancels the linear operator on \(\gamma=\alpha\). Acceptance
ladder `benchmarks/ccf_hardy_rung_acceptance.py` is vorticity end-to-end with
no multi-α collapse; CPU smoke vs `--full`. Absolute gates unchanged
(`1e-11` / `1e-6`); stretch `1e-13` is reported only when measured.
**Rung-1 / Rung-2 / stretch remain unearned until dense residual clears.**

### Added — CCF Hardy exact-Hilbert basis, whole-line CAP attempt, dissipation threshold

Line-domain CCF discovery now uses the Cauchy–Hardy pair
(`omnibias.core.verified.hardy_line`) with \(\alpha=1/(1+\lambda)\), lambda-tied
compactification, Martens–Grosse Gauss–Newton, linearized multistage, funnel,
and mpmath polish. New CAP kind `ccf_hardy_wholeline_blowup` attempts whole-line
interval covering + \(\ell^1_\nu\) NK (flips `whole_line_certified` only on
closure; sequence \(Y_0\) no longer double-counts the residual). Vorticity-form
discovery (`omnibias.pinn.jax.discovery.ccf_vorticity`) evaluates Rung-1 on a
dense Wang residual with exact \(U=H\Theta\). Absolute Rung-1 gates in
`benchmarks/_gates.py` require **both** published \(\lambda\) digits **and** the
residual gate (`reproduces_published_lambda` is not λ-alone). Acceptance ladder:
`benchmarks/ccf_hardy_rung_acceptance.py` (writes `docs/benchmarks/` only when
`earned=true`). **Rung-1 / Rung-2 are not earned yet.** A feasibility spike of
the corrected far-field / compactified-angle Hardy family (exact
\(H\Omega=U'\), mode continuation, Chebyshev+FFT PINN fallback) measured dense
Wang vorticity floors \(\approx 3\times10^{-2}\)–\(4\times10^{-2}\) under a
nontrivial gauge — still far above the \(10^{-11}\) absolute gate. Thresholds
were not weakened; Phase-2 CAP rewiring was not started. Near-null micro-scale
ghosts remain rejected by gauge/nontriviality checks. Clay NS / Yang–Mills
parents stay external.

### Fixed — causal marching seam metric + stiff reaction family

`benchmarks/causal_marching.py` now supplies the IC as `ic_fn` evaluated on the
marcher's own random `slice_points` (a linspace-ordered `ic_values` vector was
measuring a ~0.19 grid-alignment artifact, not handoff error). Hard-cage heat
seams collapse to machine zero. A second **reaction** family
(`u_t = rho u(1-u)`, `rho=12`, soft IC weight 10) gates marching beating
whole-interval -- the classical causality failure where heat-with-hard-cage
cannot show a win. Summaries surface realized `total_steps_*` because gated
retries may exceed the advertised equal budget. Schema `causal_marching/v4`.

### Added — spectral instrumentation + frontier doctrine + four-gap cookbooks

- `benchmarks/spectral_bias_fbpinn.py` records per-arm `wall_seconds` /
  `peak_rss_mb` and a parameter-matched `lstsq_matched` arm (schema v4).
- Maintainer skill `omnibias-dev-frontier-research`, rule section +
  `frontier-claims.mdc`, consumer skill `omnibias-frontier`.
- Executable cookbooks: causal marching, SDF geometry, operator zero-shot,
  one-shot least-squares.

### Fixed — `omnibias-pinn`: four-gap benchmarks now pass absolute gates

Three of four committed gap artifacts were numerically invalid: the parametric
heat reference marched with explicit RK4 (unstable above diffusivity ~0.14,
producing `max|u| ~ 1e9` that still passed `isfinite`), causal arms scored ~9×
worse than predicting zero, and spectral "beats plain" compared two failing
arms. Fixes:

1. **Heat reference** — `make_heat_slab` / `make_parametric_heat_slab` now use
   ETDRK4 (exact linear advance) plus a maximum-principle guard (torch + JAX).
2. **Absolute gates** — shared `benchmarks/_gates.py` (`rel_l2`, `skill_score`,
   validity / skill / threshold helpers); every artifact emits a `gates` block.
3. **Causality** — hard IC/BC cage + closed-form residual + `advance_policy="gate"`.
4. **Geometry** — interior Poisson solve with manufactured `u* = |φ|·…` vs soft
   penalty (boundary identity kept as by-construction).
5. **Spectral** — one-shot frozen-feature least-squares arm dissolves GD spectral
   bias; capacity falsification at high frequency.
6. **Operators** — shared-IC diffusivity sweep (conditioning necessary);
   width-1 parameter heads use `Identity` instead of LayerNorm (which mapped
   every scalar to 0); comparator is a residual PINN (IC + closed-form heat
   residual), not full-field supervised fit.
7. **Doctrine** — discovery framing in rules/skills; CI wires the four smoke
   benchmarks into the `pinn` job; smoke writes `*_smoke.json` so CI cannot
   clobber multi-seed acceptance artifacts.

### Added — `omnibias-pinn`: four-gap PINN closure (`train` / `domain` / operator / spectral)

Acceptance-gated alpha capabilities folded into Beta `omnibias-pinn`. Claims are
earned per tested PDE / domain family once the absolute gate passes
(`docs/benchmarks/pinn_four_gap_matrix.md`).

1. **`omnibias.pinn.train`** — `march_solve` (torch + JAX) with advance gating,
   required IC, same-time triviality guards, optional `hard_ic_factory`,
   per-component residual RMS, seam diagnostics; `SpectralBandScheduler` applies
   measured residual bands at deterministic steps.
2. **`omnibias.pinn.domain`** — negative-inside R-function CSG, boundary-factor /
   jet protocol, SDF-driven solver sampling, `DistanceConstrainedField` with
   Dirichlet / Neumann / Robin modes (explicit junction failure for normals).
3. **Operator conditioning** — multi-head encoders + fusion (function /
   parameters / BC / geometry; width-1 parameter heads skip LayerNorm), ETDRK4
   parametric slabs, geometry hard-BC wrapping, JAX FNO2d twin; zero-shot bench
   vs unconditioned + per-instance residual PINN retrain under equal budget.
4. **Spectral bias** — mutation-free NTK + Lanczos path, multilevel FBPINN,
   one-shot least-squares arm, mode-wise equal-param arm bake-off.

Guarantee levels: hard cages / window geometry / ETDRK4 linear advance are by
construction; training and zero-shot accuracy are empirical (multi-seed
`--full`) with absolute skill floors; certificates remain a-posteriori residual /
linear-PDE scope. API pages `pinn-train` / `pinn-domain`, examples
`pinn_causal_marching` / `pinn_sdf_geometry` / `pinn_fbpinn`, smoke benchmarks
`causal_marching` / `geometry_sdf` / `operator_zero_shot` /
`spectral_bias_fbpinn`.

### Added — `omnibias-pinn`: neural operator learning (`omnibias.pinn.operator`)

DeepONet / FNO operator learning as an alpha submodule of `omnibias-pinn`. A
DeepONet is linear in its trunk basis, so every query-coordinate mixed partial of
`G(u)(y)` is a closed-form trunk jet times the branch coefficients -- one jet
yields a full PDE residual mesh-free, with no finite differences and no
periodic-grid requirement on the query side. FNO is shipped as the honest
FFT-based / periodic baseline; its derivatives are not closed form. A residual
enclosure over a branch-coefficient box is sound interval arithmetic, not a
solution-error bound; operator accuracy is optimised, not proven. Naming sense 3
of `docs/operator-surface.md` (not `OperatorBlock`, not a field operator).

Torch + jax twins (`build_deeponet` / `make_deeponet`, heat / Burgers / KS slabs
and residuals including `ks_residual_loss_fd`), verified residual helpers with a
sensor-box → coefficient-box → sealed residual family certificate,
`docs/examples/pinn_operator_learning.py` (order-4 exactness + KS one-jet +
FD-floor smoke + family cert), API page `docs/api/pinn-operator.md`, CPU smoke
benchmarks (`operator_fd_floor` / `operator_shared_grid` /
`operator_residual_calibration` / Burgers `operator_deeponet` / KS
`operator_ks_bakeoff` / `operator_fno_vs_deeponet`), and a CI smoke step. The
authored-strict operator modules are on `scripts/mypy_strict_allowlist.txt`.

Measured bake-offs (decision rules fixed before the runs; 8 seeds): Burgers is
an honest negative — closed-form residual does not beat FD-`u_t` on median
held-out rel-L2 (FD error at `dt=0.05` is ~1e-8). KS closed-form beats
FD-`u_xxxx` on median but only 5/8 seeds (seed-fragile; rule required ≥6/8).
Structural claims (order-4 closed form, one-jet residual, measured FD floor
4.5e-6) stand regardless.

### Changed — `omnibias-pinn`: the periodic-seam claim is scoped to the orders it matches


No behaviour changed here; what changed is what the benchmark can prove and what
the docs are entitled to claim. The hard-conditions boundary-violation column
scores exactly the orders each condition declares. For Dirichlet and Neumann that
is the whole condition. For a periodic seam it was graded on its own syllabus --
the cage enforces `PERIODIC_ORDERS` and the column measured those same orders --
so it could never report a kink one order up, and "exact seam" was doing more
work in prose than the measurement supported.

`benchmarks/hard_conditions_solver.py` (schema v3) now also probes the first
*unmatched* order, by re-declaring the seam there and reassembling, so the probe
travels the same row path as the enforced orders instead of a lookalike that
could drift from it. Measured: the `(0, 1, 2)` cage jumps `6.1e+00` at order 3 on
the gauge-free seam, 2.5% of that derivative's own scale, and `2.6e+01` on the
gauge-pinned heat, 11.1%. Two results fall out of the new column. The hard arm is
better behaved than soft *beyond* its contract as well as inside it (11% against
64%), so absorption is not buying its exactness by displacing error just past
where anyone was looking. And the **spectral arm is the only one that closes the
seam at every order** (exactly `0.0`), because periodicity there is a property of
the Fourier basis rather than three constraints bolted onto a generic one -- an
advantage the interior-L2 column does not show at all. Both cage suites now pin
the contract boundary in torch and jax: machine-zero at every declared order,
genuinely discontinuous one order up, with three orders of magnitude of margin.

`benchmarks/hard_conditions_periodic_sweep.py` (schema v2) gained `(0, 1, 2, 3)`
and a stopping rule fixed before the run: raise the default again only if the
fourth matched order buys more than the third did. It does not -- read on the
hard arm alone, since the hard−soft gap conflates it with the soft arm getting
worse as rows are added, the third order gains 3.34x and the fourth 1.30x. **The
default stays `(0, 1, 2)`**, now resting on an exhausted sweep rather than a
two-point one. A smooth manufactured solution matches every derivative and so
rewards extra orders indefinitely, which is exactly why diminishing returns are
the signal to stop rather than continue; and a higher default would over-smooth
seams near steep gradients, the failure the periodic-emit measurement already
recorded for Burgers.

### Fixed — the release guards job was failing on a guard that never existed

`.github/workflows/ci.yml` invoked `packages/omnibias-core/tests/test_no_leakage.py`
by path in the `guards` job, and `AGENTS.md`, `GOVERNANCE.md`,
`docs/release-readiness.md` and `docs/release_blockers.md` all described it as the
enforcing guard for the vendor-neutral-language rule. It had never existed in the
repository's history. pytest aborts on a missing path argument without running
anything, so the job did not merely skip that one check -- the placeholder,
version, terminology, licence, lineage, sorted-`__all__`, resolvable-`__all__` and
`py.typed` guards in the same invocation never ran either, and vendor neutrality
was enforced by nothing.

The guard now exists and blocks seven leak families across every file in the
working tree -- tracked files *and* new ones that are not gitignored, which is
slightly wider than the rule as written and deliberately so, since a leak is
easiest to introduce in a file that has just been created and a guard that waits
for the commit reports it a step too late (reaching `.github/`, `notebooks/` and
`formal/`): local developer paths, site
mounts and scratch conventions, batch-scheduler commands / script directives /
environment variables / product names from any vendor, the EDA-shop synonym for a
compute cluster, personal accounts, private-tree paths, and credentials. It cannot
go vacuous: every pattern carries synthetic bait a self-test asserts it catches,
the scanner is driven end-to-end over that bait, and the file list is pinned as
large and as actually reaching those trees. Sanctioned vocabulary is pinned as
legal in the same test -- `$OMNIBIAS_SCRATCH`, "GPU job" / "GPU cluster", the
published Derivon identity, and the prose that *documents* the rule -- so the
guard cannot tax good writing. The tree needed no redactions; it was already
clean.

Restoring the job immediately caught two unsorted `__all__` lists,
`pinn._core.constrained` and `pinn.solver._core.hard`, both extended during the
hard-conditions work. Both are sorted again.

### Fixed — `omnibias-pinn`: jet-field value path no longer pays for a jet

`JetMLPVectorField.value_component` (torch + jax, inherited by attention /
multiscale) again takes the plain forward pass. The readout-independence refactor
had routed values through `_jet_at_least(..., 0)`, which builds at
`max(0, jet_order)` and made every boundary-condition loss cost a full order-2
(or order-3) jet. After the hidden/readout split both paths already read live
parameters, jet fields are refused by the frozen-feature linear solver, and the
forward / row-0 equality is pinned at `1e-14`, so the cheap path is the correct
one. A regression test asserts a value-only evaluation leaves the hidden-jet
cache empty.

### Added — `omnibias-pinn` / `omnibias-fields`: readout-independence invariant + spectral linear solve

The frozen-feature linear solver reuses cached `FieldState`s while sweeping the
readout. That is sound only when every cached quantity is independent of those
weights. The contract is now named and enforced:

- `READOUT_INDEPENDENT_ATTR` (`"_omnibias_readout_independent"`) on
  `omnibias.fields` / `omnibias.pinn.solver` -- fields that honour the contract
  declare it; the driver raises `ReadoutDependentError` otherwise.
- Spectral / Chebyshev caches store the hidden temporal features `h` (not
  `a = b_t + h @ V.T`); jet fields cache the hidden jet through `layers[:-1]` and
  apply `affine_jet_mv` per call, so a state stays coherent under a readout
  sweep.
- Affine cages recurse through `base` and declare when the base does.
  Nonlinear cages (`IntegralConservationField`, `NormConservationField`) decline
  and are refused by `solve_least_squares` -- use `solve_optimize` instead.
- Per-backend readout seam (`readout_size`, `set_readout` / `with_readout`,
  `readout_dtype`) replaces hardcoded `W` / `c` access, and
  `build_field` / `solve_least_squares` accept `basis="spectral"` (requires a
  time axis; spatial periodicity is free in the Fourier ansatz).

### Changed — `omnibias-pinn`: `PERIODIC_ORDERS` default is now `(0, 1, 2)`

A periodic `BoundaryCondition` carries `periodic_orders` (default
`PERIODIC_ORDERS`). The Stage-4 C¹-seam sweep
(`benchmarks/hard_conditions_periodic_sweep.py`) showed that matching only
value and slope leaves a second-order operator free to jump in `u''` across a
gauge-free Poisson seam; raising the matched orders to `(0, 1, 2)` flips hard
vs soft interior L2 in hard's favour on every seed. Override with an explicit
tuple if you need the old subset.

The hard-conditions benchmark also gained a **gauge-pinned** periodic heat row
(hard wins 5/5, absorbed 4 under the shipped `(0, 1, 2)` default) and a third
**spectral** arm (`basis="spectral"`, `K=8`): machine-zero seam residual, beats
soft on every seed, does not yet beat the MLP cage on that budget. Regenerated
under the same default, the gauge-free Poisson seam row also wins 5/5 (absorbed
3; interior `3.6e-05` vs soft `5.7e-05`) -- a flip from the earlier `(0, 1)`
measurement that lost on every seed.

### Added — `omnibias-pinn`: opt-in `periodic_boundary` on the six problem builders

The six canonical builders (`poisson`, `heat`, `wave`, `burgers`,
`reaction_diffusion`, `advection_diffusion`) accept
`periodic_boundary: bool = False`. When `True`, each emits one periodic
`BoundaryCondition` per component per periodic *spatial* axis, **appended**
after existing BCs so `absorbed_boundary` indices stay stable. The default is
off and reproduces today's boundary tuples / residual arrays bit for bit
(`packages/omnibias-pinn/tests/solver/test_periodic_boundary_emit.py`).

Measured (`benchmarks/hard_conditions_periodic_emit_measure.py`, artifact
`docs/benchmarks/hard_conditions_periodic_emit_measure.json`): manufactured
periodic Burgers and reaction-diffusion, 3 seeds, hard and soft, identical
small budget, under the shipped `PERIODIC_ORDERS = (0, 1, 2)`. Decision rule
stated up front — flip the default to `True` only if it is strictly better on
both interior rel-L2 and seam violation for both problems. **Default stays
`False` (opt-in):** reaction-diffusion wins on both metrics with emit on
(interior `4.5e-02` vs `5.6e-02`, seam `~9e-16` vs `7.3e-01`), but Burgers
closes the seam (`~2e-16` vs `1.0`) while the interior fit gets worse
(`1.5e-01` vs `2.3e-02`). Absorbed counts under emit-on / hard are 4 (Burgers)
and 8 (reaction-diffusion).

### Added — `omnibias-pinn`: hard Neumann / Robin / initial conditions, with solver auto-detection

The only hard-condition cage the package shipped was `HardBoundaryField`, whose
multiplicative ansatz `u = g + d * f` handles arbitrary geometry but **only
Dirichlet data**, and does not compose: wrap it around a derivative condition and
the distance factor lands on that condition, breaking it. Everything else — a
Neumann flux, a Robin law, an initial state, an initial velocity — was left as a
penalty term, weighted against the interior residual by a `condition_weight` the
user had to tune.

The *additive* switching form composes exactly, which is what makes the general
engine no harder to build than the special cases. For linear conditions
`C_k[u] = t_k`, pick support functions `s_j`, form the support matrix
`M_kj = C_k[s_j]`, and set `phi_i = sum_j (M^-1)_ji s_j` so `C_k[phi_i] =
delta_ki`. Then `u = g + sum_k phi_k (t_k - C_k[g])` satisfies every condition
for **any** free function `g` — the Theory of Functional Connections (Mortari
2017; Leake and Mortari, *Mathematics* 8(8):1303, 2020). Applying it once per
axis embeds conditions on several axes at once, and the cross terms that make
the corners come out right are generated by the recursion rather than
special-cased.

- `omnibias.pinn._core.constrained` holds the algebra in pure Python —
  `LinearConstraint`, `AxisConstraints`, `support_matrix`,
  `switching_coefficients` — so both backends read one source and cannot drift.
  Support degrees are chosen by **rank**, not assumed to be `0..n-1`: a pure
  Neumann set annihilates the constant, which is a well-posed problem the naive
  monomial family cannot see.
- `ConstrainedExpressionField` (torch + jax twins, parity-tested at `1e-12`) is
  closed form *in the network*: every value and derivative of `g` it needs,
  including at the projected face points, comes from the sigma-tower. Autodiff
  appears only when a user-supplied target callable has to be differentiated
  along another axis. It recurses over **every** constrained axis, so the edges
  and corners where two or three axes meet come out exact without a special
  case. Cost, reported by `projection_cost` rather than left implicit, is the
  product over axes of `1 + #distinct projection points`: a face carrying both a
  value and a slope costs one, but a second constrained axis multiplies rather
  than adds, which is why absorbing every face of a 3-D box is a choice.
- Periodicity is absorbed as a **relative** constraint `d^n u(hi) - d^n u(lo) =
  0` for each `n` in `periodic_orders` (default `(0, 1, 2)`: value, slope, and
  second derivative), matching a linear functional that may reference several
  points. This also fixes a real gap: the solver previously
  emitted a zero-length row for a periodic boundary on the grounds that the
  ansatz carried it, which is true of a spectral method-of-lines discretisation
  and false for the mesh-free route. The soft path now assembles genuine seam
  rows, so `hard_conditions="none"` enforces periodicity approximately where it
  previously did not enforce it at all.
- `plan_hard_conditions(system)` triages a solver `System` into what can be
  absorbed and what cannot, with a reason for every decline, and
  `hard_conditions="auto"` on `solve_least_squares` / `solve_optimize` /
  `solve_inverse` (plus the jax least-squares twin) builds the ansatz and drops
  those rows from the loss. Absorption is **partial** by design: whatever is
  provably exact becomes structural, everything else stays soft. The plan object
  is the single source of truth for both halves, so the field and the loss can
  never disagree about what is enforced.

Two preconditions are checked rather than assumed, and both are live falsifiers
in the tests rather than assertions in prose. `certify_support_matrix` builds `M`
entrywise in outward-rounded interval arithmetic, encloses
`lambda_min(M^T M) > 0` (the Gram, since `M` is not symmetric), widens it by the
Weyl bound so the claim is about the *exact* Gram rather than its float image,
and seals a hash-verifiable certificate — a finite rational obligation, so it is
in scope for the Lean kernel. A dependent condition set is **refused**, not
approximated. Separately, condition data on different axes has to agree where
those axes meet, and construction now refuses data that does not, naming the two
conditions that clash. The gate is checked over **every pair** of axes rather
than consecutive ones — quadratic in the conditions, not exponential in the axes
— on a deterministic Kronecker lattice, so a refusal is reproducible across
backends rather than seed-dependent. `compatibility_residual` keeps the
disagreement visible at its true order-one size, including the case where the
values agree perfectly and only the slopes clash (a time-dependent Dirichlet
against a zero initial velocity), which is easy to write by accident and
impossible to see by inspection.

The guard that matters most is the one against silence: after an auto-caged
solve the tests re-assemble the **full** condition residual *ignoring* the plan's
absorption and assert it is below `1e-12`. If the plan ever claimed a condition
the cage does not enforce, the loss would stop watching it and nothing else would
notice. `hard_conditions="none"` is the default and reproduces the previous solve
bit for bit, pinned by its own test.

Measured (`benchmarks/hard_conditions_solver.py`, artifact
`docs/benchmarks/hard_conditions_solver.json`): Poisson / heat / wave / a 2-D
square / a periodic seam / a gauge-pinned periodic heat, 5 seeds, identical
architecture, parameter count, seed and collocation budget, under the shipped
`PERIODIC_ORDERS = (0, 1, 2)`. The hard arm's worst boundary violation over
every cell was `1.3e-13`, and its median interior relative L2 beat the soft arm
on Poisson (`3.5e-07` vs `1.6e-06`), heat (`3.8e-06` vs `1.1e-02`), wave
(`1.1e-06` vs `3.1e-03`), the square (`2.8e-05` vs `7.3e-02`), the gauge-free
seam (`3.6e-05` vs `5.7e-05`) and the gauge-pinned heat (`2.4e-04` vs
`4.2e-01`), 5 seeds out of 5 on all six. The parabolic and hyperbolic gaps are
the large ones because that is where the soft arm has an initial condition
competing with the interior residual. Under the earlier `(0, 1)` default the
gauge-free seam had *lost* ~3x on every seed; the C¹-seam sweep implicated
discontinuous `u''` rather than lost degrees of freedom alone and flipped
`PERIODIC_ORDERS` to `(0, 1, 2)` (see the Changed entry above), after which the
regenerated table shows hard winning. Absorption buys a guarantee, and under
C² matching it is no longer free of accuracy either.

Scope, stated rather than implied. The domain must be an axis-aligned box;
`HardBoundaryField` remains the route for arbitrary geometry, and the docs now
say which to use when instead of describing it as the only hard-BC cage.
Conditions on any number of axes are in scope, but a condition whose support
matrix will not certify is declined with a reason and stays soft, which is what
the solver does today. The conditions are exact by construction; interior
accuracy is optimised, not proven.

`docs/examples/pinn_hard_conditions.py` is runnable and wired into CI.

### Added — `omnibias-pinn`: conservative shock capturing (cage over partition)

Representing a shock wants a sharp seam; representing a conservation law wants
`div G = 0`. The two were separately available and had never been composed,
because the obvious composition is wrong: making each partition patch a
`FluxFormField` gives `div (sum_l w_l G_l) = sum_l grad w_l . G_l`, which
vanishes only where the gates are saturated — conservation breaks exactly at the
seam, the one place a shock needs it.

Inverting the nesting fixes it, and needed no new code. `FluxFormField` builds
`G^i = sum_j d_j P^{ij}` from an antisymmetric potential, so
`div G = d_i d_j P^{ij} = 0` by symmetry of mixed partials, for *any*
twice-differentiable `P` however sharp. Putting the partition inside, as the
potential, buys an arbitrarily sharp front at no cost to the conservation law.
That is also the right representation rather than a convenient one: with axes
`(t, x)` the cage gives `rho = d_x P`, so `P` is the cumulative mass, and the
exact viscous profile integrates to `P -> c0 x - (c0^2 + a^2) t/2 - a |x - c0 t|`
as `nu -> 0`. The potential of a shock is a **kink**, which is what a partition
of unity over smooth patches already represents well.

- `packages/omnibias-pinn/tests/partition/test_cage_over_partition.py` pins the
  composition: relative `|div G| < 1e-12` for `beta` from 2 to 200 (measured
  2e-16 to 3e-14), holding after training relocates the seam, and the rejected
  nesting is kept as a live falsifier at order-one divergence — without it the
  passing test would look like a property of cages in general.
- `docs/examples/pinn_burgers_shock.py` and
  `benchmarks/burgers_shock_conservation.py` (artifact:
  `docs/benchmarks/burgers_shock_conservation.json`) train the cage against a
  non-conservative arm at identical architecture, parameter count, seed and
  collocation budget, over 6 viscosities x 5 seeds.

Measured, and reported the way it came out. The conservative arm holds global
mass balance tighter at **every** viscosity, and the margin widens as the layer
goes under-resolved: 1.3x at `nu = 2e-2`, 2.9x at `nu = 2e-3`, winning 5 seeds out
of 5 once `nu <= 3e-3`. As `nu` falls tenfold the baseline's mass error grows
4.5x while the cage's grows 2.0x. On shock speed and relative L2 the ordering
**reverses** with resolution — the non-conservative arm is up to 16x better where
the layer is resolved — so only the mass-balance result is asserted in CI and the
rest is printed. That is the finite-volume result carried onto a mesh-free field:
a conservative scheme is not more accurate on a well-resolved smooth problem, it
is more robust when the feature is under-resolved.

Scope: `div G = 0` is structural (3.4e-15 worst over all 30 sweep cells, no
training, no quadrature, no tolerance); solution accuracy is optimised, not
proven. Conservation pins the Rankine-Hugoniot jump condition and does **not**
select the entropy solution. The `beta -> inf` seam sharpening is *temperature
collapse*, never the founding `delta -> 0` bias collapse.

Also adopts `docs/examples/partitioned_pinn.py`, which existed but appeared in no
`docs/examples.md` row, no `llms.txt` entry and no CI job, so nothing had been
running it.

### Fixed — docs: the published derivative-order ceiling understated the tower

`docs/scope-and-guarantees.md` sec 2 capped `huber`, `silu`/`swish`, `gelu`,
`relu` and `mish` at order 1, blaming a "Dirac at the kink". Every entry in that
row has had an all-orders fast path for some time, and three of the five have no
kink at all: `silu`, `gelu` and `mish` are smooth everywhere and carry exact
Leibniz towers over the `z f(z)` product, while `relu` and `huber` carry
all-orders almost-everywhere / regular-part towers. `smooth_sign` was likewise
listed at 2 despite being `tanh` rescaled, hence unbounded.

The table is labelled "reading guidance for AI agents", so a stale ceiling there
does not merely go out of date — it teaches every agent and reader the wrong
answer. `tests/test_doc_activation_orders.py` now parses the table and checks
every claim against the live registry on both backends, including a negative half
that requires each claimed cap to genuinely raise past its ceiling, so replacing
the table with "unbounded" everywhere would fail. Also corrects the stale header
tables in `omnibias.torch.activations.nqs` (`mish`, `smooth_sign`) and a section
comment in `omnibias.jax.activations`.

### Added — `omnibias-geometry`: certified lattice mass gap (`omnibias.geometry.gauge.transfer`)

The rigorous gap engines in `omnibias.core.verified.eig` were written *for* this
application — their docstrings name heat-kernel and Wilson transfer matrices, the
`+/- n` U(1) modes and the `(p,q) <-> (q,p)` SU(3) pairs — and were then never
connected to one. This connects them.

- **Transfer matrices** (`omnibias.geometry.gauge.transfer.matrices`):
  `u1_heat_kernel_transfer` (`character` diagonal or `angle` dense circulant),
  `su2_heat_kernel_transfer` / `su3_heat_kernel_transfer` (eigenvalues
  `exp(-t C2)` from the *exact* `Fraction` of `quadratic_casimir`),
  `su2_class_angle_transfer` (the `su(2)` spectrum in a dense, entrywise-positive
  basis that a Markov chain can actually move in), and `su2_wilson_transfer`
  (character expansion via the new `besseli_iv`). Entries are outward-rounded
  intervals; the closed-form spectrum is carried alongside, so a certified bound
  can be checked against truth rather than believed.
- **Certified gaps** (`.gap`): `certified_transfer_matrix_gap` dispatches to the
  symmetric power-sum engine with a partner chain, or Birkhoff-Hopf, whichever is
  applicable and tighter, keeping every candidate it considered.
  `certified_multistep_gap_refinement` sharpens via `T^n`;
  `certified_effective_mass_curve` supplies rigorous *upper* bounds so the true
  gap is genuinely sandwiched; `heat_kernel_gap_scaling_report` records bounds
  across spacings as evidence about a trend.
- **Certificates and a registry** (`.certificates`, `gauge/proofmachine.py`):
  sealed, tamper-evident `verified-transfer-matrix-gap-1` certificates carrying a
  top-level `subdominant_ratio_upper`, so the Mathlib-free Lean kernel's
  `spectral_gap_pos` lemma discharges the obligation with no new Lean. Replay
  rebuilds the matrix from its recorded *constructor arguments* and rejects a
  sealed bound tighter than an independent derivation supports. `gauge_provers()`
  / `build_gauge_machine()` mirror `omnibias.sos.proofmachine`, since
  `omnibias-pinn` does not depend on `omnibias-geometry`.
- **Monte-Carlo cross-check** (`.montecarlo`): rather than assume a matrix
  corresponds to an ensemble, `certified_gap_versus_monte_carlo` samples the path
  measure `prod_t T_{x_t, x_{t+1}}` that the matrix *itself* defines, reading
  matrix entries only. On `su(2)` the certified bound is exactly tight and the
  sampled effective mass brackets the closed-form gap.

Scope, unchanged and non-negotiable: every certificate is a statement about **one
fixed matrix at one fixed spacing in finite dimension**. `continuum_claim` is
hard-wired `False`, the scaling report is labelled evidence, and nothing here is a
claim about the Yang-Mills mass gap.

### Added — `omnibias-core`: `besseli_iv`

`omnibias.core.verified.besseli_iv` encloses the modified Bessel function
`I_n(x)`, following the existing `erf_iv` mpmath-bracket pattern so it inherits
`strict_backend()` / `libm_fallback_used()`. Because
`I_n(z) = sum_k (z/2)^(2k+n) / (k! (k+n)!)` is all-positive, the no-mpmath path is
a truncated series with a rigorous geometric tail bound — unconditionally sound,
not a ulp-inflated guess — and refuses arguments too large to bound soundly.

### Fixed — `omnibias-pinn`: `perron_spectral_gap` could never reach the Lean kernel

`_perron_certificate` never called `seal_certificate()`, while `check_certificate`
refuses unsealed certificates before emitting any Lean. `generate_obligation`
succeeded (the math was Lean-ready) but `verify_certificate_digest` was always
`False`, so the kind could never earn `theorem_prover_verified`.
`test_perron_lean_check_flag_mirrors_kernel` passed only because no runner had
Lean installed, and would have failed the moment one did. The certificate is now
sealed, its schema validates the digest, and a **generic guard test** asserts that
every default-machine prover whose certificate yields a non-`None`
`generate_obligation` also passes `verify_certificate_digest`, so this class of
bug cannot recur silently.

### Fixed — `omnibias-core`: the `Prover` protocol demanded a settable `name`

`Prover` declared `name: str`, which requires a *mutable* attribute, so the repo's
own frozen `FunctionProver` did not satisfy its own protocol under a strict type
check. It is now a read-only property, which accepts both plain attributes and
properties. No caller assigned through a `Prover`-typed reference, so this is
backwards-compatible.

### Added — `omnibias-pinn`: deep fields, multi-scale, balancing, conservation, decomposition

Before this change the only trainable free-form PINN field on the substrate was
`OneLayerVectorField` — a *single* hidden layer. `JetMLP`, `FourierFeatureMLP`,
and `make_siren` already existed in `omnibias.{torch,jax}.architectures` with
the closed-form tower intact, but returned raw tensors, so they could not reach
the field operators, the cages, or the prebuilt PDE residuals. They now can.

- **Deep fields** (`omnibias.pinn.{torch,jax}.fields`): `JetMLPVectorField`,
  `FourierFeatureVectorField`, `make_siren_vector_field`, and
  `build_jet_mlp_vector_field`, on a new `jet_mlp` dispatch tag. Every partial
  is read off **one** memoised multivariate jet (`state.extra`, keyed by total
  order) rather than recomputed per axis, so an order-2 Navier–Stokes residual
  costs one order-2 jet for the whole residual; `gradient_full` / `hessian_full`
  / `laplacian` take that fast path. A `depth=1` `JetMLPVectorField` reproduces
  `OneLayerVectorField` derivatives exactly.
- **Multi-scale** (`omnibias.{torch,jax}.architectures.multiscale` +
  `MscaleVectorField` / `AdaptiveJetMLPVectorField`): `AdaptiveActivation` is
  the Jagtap adaptive activation `sigma(n a z)` built from the new backend-
  neutral `omnibias.core.spec.tempered` combinator, so a *trainable* frequency
  still gets the whole tower `(n a)^k sigma^(k)(n a z)` for free rather than a
  hand-written derivative. `MscaleMLP` is the MscaleDNN band mixture
  `u = sum_j f_j(alpha_j x)`, one exact jet per band. `suggest_frequency_bands`
  closes the loop from the existing `power_spectrum_per_d` / `spectral_fidelity`
  diagnostics back into band selection.
- **Loss balancing** (`omnibias.pinn._core.weighting` +
  `omnibias.pinn.{torch,jax}.losses.weighting`): a *stateful* `LossWeighter`
  with EMA and update cadence, gradient-norm annealing, and self-adaptive
  pointwise weights trained by gradient **ascent**. Today's `ntk_balanced_loss`
  is stateless and recomputed from scratch; these carry state across steps.
- **Causal time-marching** (`omnibias.pinn._core.marching`):
  `TimeWindowSchedule` turns `causal_residual_loss` into real marching — time-
  binned collocation sampling, warm start of window `k+1` from window `k`, a
  causality-tolerance advance criterion, and epsilon annealing.
- **Conservation cages** (`omnibias.pinn.{torch,jax}.cage`):
  `IntegralConservationField` generalises `omnibias-qpinn`'s
  `NormConservationField` into a domain-neutral cage holding
  `int sum_c u_c^p dx = C` by quadrature rescaling — exact at every optimiser
  step, including step 0, to quadrature accuracy. `FluxFormField` writes a flux
  as `G^i = sum_j d_j A^ij` with `A` antisymmetric, making `div G = 0` an
  algebraic identity rather than a penalty.
- **Non-local field**: `AttentionVectorField` (and `AttentionJetMLP` in
  `omnibias.{torch,jax}.architectures`) — a softmax mixture over a trainable
  memory whose *coordinate* derivatives are closed form to arbitrary order.
  `omnibias.hopfield` differentiates the same block with respect to the
  **scores**; a PDE needs `d/dx`, which is what the new jet primitives supply.
- **Domain decomposition** (`omnibias.pinn._core.interface` +
  `omnibias.pinn.{torch,jax}.losses.interface`): `Interface` / `InterfaceSpec`
  geometry with `interface_points` and `split_by_interface`, and the XPINN /
  cPINN residuals `value_jump`, `flux_jump`, `normal_derivative`,
  `interface_residual`, `interface_loss`. The seam sampler draws points **on**
  the interface in its own tangent coordinates rather than near it — the
  hand-rolled `10.0 * interface` penalty the tests used before measures the jump
  plus a discretisation error, and no amount of training removes the second
  part. The flux condition carries the material pair `(k_+, k_-)`, so a
  conductivity contrast is expressible at all. Everything reads the field
  through `state.ops`, so either side may be any field type.
- **Heterogeneous patches**: `PartitionedField` (torch and jax) accepts a
  `Sequence[FieldBase]` of *different* field types and sizes instead of forcing
  identical `OneLayerVectorField`s, validating that they agree on the coordinate
  and component specs. `build_partitioned_field` takes per-region `hidden` /
  `base`, or a `subfield_factory(region) -> FieldBase` for full control. Both
  backends' `OneLayerVectorField` gained `forward_values`, the one-line contract
  a composite needs from a sub-solution.

### Added — `omnibias.pinn.solver`: stiff integrators

`omnibias.pinn.solver.{torch,jax}.stiff`, bit-identical twins. Until now the
only implicit step was `implicit_linear_step`, which handles a linear *diagonal*
Fourier symbol on a periodic 1-D grid; everything else was explicit and capped
by the fastest decaying mode. Four families now cover the rest, all written as
ordinary differentiable functions so a step composes inside a training graph:
`rosenbrock_step` (ROS2, L-stable, one LU and two solves),
`exponential_rosenbrock_step` (exact on an affine right-hand side, at any step
size), `imex_euler_step` / `imex_cnab2_step`, and `etdrk4_step`.

- `phi_diagonal` / `phi_matrix` evaluate `phi_k` by a scaled Taylor series and
  exact doubling identities rather than the defining quotient, which cancels
  every significant digit as `z -> 0`: `phi_1(1e-14)` comes out as `1 + 5e-15`
  where `(exp(z) - 1) / z` is wrong in the fourth digit. Complex symbols work,
  since a Fourier-space `L` is one.
- `closed_form_jacobian` reads a stiff step's linearisation off **one** order-1
  multivariate jet of the `(W, b, spec)` layer stack — no autodiff graph, no
  finite difference. `dense_jacobian` is the honest autodiff fallback for an
  arbitrary callable.
- `SemiDiscrete` now carries `nonlinear` alongside `symbol`, so the linear /
  nonlinear split an IMEX or ETD scheme needs is stated once and `method_of_lines`
  accepts `"etdrk4"`, `"imex_euler"`, and `"imex_cnab2"`.
  `kuramoto_sivashinsky_semidiscrete` is the canonical stiff case; `stiff_rollout`
  composes any step into a differentiable trajectory.

### Added — `omnibias-torch` / `omnibias-jax`: non-elementwise jet algebra

`jet_reciprocal`, `jet_exp`, `jet_softmax`, and `jet_attention` in
`omnibias.{torch,jax}.jet_mv`, bit-identical twins. Until now the multivariate
jet chain was elementwise-only; these close it under division, exponentiation,
and normalisation, which is what lets an attention block sit inside
`mlp_jet_mv` with the tower intact. `jet_exp` reuses the `exp` activation's
derivative tower rather than forking a second one, and `jet_softmax` is
max-shifted for stability with the shift cancelling exactly in every
higher-order coefficient.

### Fixed

- **Documented API that never existed.** `docs/api/pinn.md` listed 21
  `omnibias.pinn.certified` members with no implementation behind them — a
  candidate-discovery-sprint surface (`candidate_family_catalog`,
  `run_fast_candidate_sprint`, and the per-stage functions around them) and a
  certified transfer-matrix / heat-kernel spectral-gap surface. The
  `navier-stokes-certified` cookbook additionally *ran* the sprint API in a
  fenced block. Since the package genuinely does not ship those capabilities,
  the claims are withdrawn rather than back-filled: documenting an unimplemented
  Navier-Stokes blow-up candidate pipeline is exactly the kind of capability
  assertion the honesty gates elsewhere in that module exist to prevent. All of
  `docs/api/` is now audited to zero undefined members.
- **`omnibias-fields` tests were not hermetic in dtype.** `test_finite_strain`,
  `test_mhd`, and `test_kinetic` each called `torch.set_default_dtype(float64)`
  at *import* time. That is a process-global mutation applied during collection,
  so omnibias-torch's own autouse dtype fixture would restore float32 before the
  fields tests ran, and five elasticity tests failed on mixed-dtype `einsum`
  whenever the two suites shared a pytest session. The default now comes from an
  autouse fixture in the fields `conftest.py`, matching the omnibias-torch
  precedent, so it holds per test regardless of collection order.

## [0.4.0] - 2026-08-04 — initial public release

First public release. Everything below is new to the outside world; nothing was
published before this tag, so there is no upgrade path to describe and no
deprecation to observe.

> This file starts here on purpose. Development happened in a private tree, and
> a changelog of pre-release churn against changes nobody could have depended on
> is noise, not history. From this entry forward, every behavioural change is
> recorded.

### The primitive

omnibias computes the **closed-form n-th derivative of an activation**,
`sigma^(n)(z)`, for arbitrary `n`, with bit-stable accuracy and a **single
`sigma` evaluation regardless of order**. The polynomial coefficients come from
one shared pure-Python module, so PyTorch, JAX, and Keras 3 are **bit-identical
by construction** rather than by test.

The math is the Riccati identity (`sigmoid' = s(1-s)`, `tanh' = 1 - t^2`) and
the Eulerian / Legendre / Hermite recurrences it implies, extended to deep
compositions and mixed partials by Bell / Faà di Bruno combinatorics.

`OperatorBlock` dispatches six roles — `identity`, `grad`, `laplacian`,
`derivative`, `band`, and `integral`. The `integral` role is a genuine
closed-form antiderivative window, `S(z+b_hi) - S(z+b_lo)` with `S' = sigma`,
not a quadrature.

### Packages

42 distributions, versioned independently.

**Stable core** — `omnibias-core` 0.4.0 (pure-Python math: polynomial
coefficients, Bell / Faà di Bruno, multi-index jets, the rigorous
`omnibias.core.verified` substrate and the `omnibias.core.proof` certificate
format), `omnibias-torch` 0.4.0, `omnibias-jax` 0.4.0, `omnibias-ferminet`
0.2.0.

**Beta** — `omnibias-fields` 0.1.0 (the field substrate: `FieldState`, views,
`SigmaCache`, and the torch / jax differential-operator surface),
`omnibias-geometry` 0.2.0 (metric, Christoffel, covariant derivative,
Laplace–Beltrami, curvature, geodesics, exterior calculus, gauge theory),
`omnibias-pinn` 0.1.0, `omnibias-symbolic` 0.1.0 (neural-jet equation
discovery), `omnibias-fractional` 0.1.0.

**Alpha** — 33 further packages spanning quantum PINNs, curvature and
second-order optimisation, measure integration, score / SDE, variational
calculus, the `delta -> 0` difference register, q-calculus, time-scale calculus,
holonomic / D-finite methods, quantisation, Boolean algebra, spiking neurons,
modern Hopfield attention, certified verification, validated dynamics, SOS
positivity, the Mathlib-backed formal checker, differentiable discrete
optimisation (QUBO / submodular / structured DP / tabular / partition /
combinatorics / NP-hard families / routing / logic), convex programming,
spectral graph operators, control, shape fields, and the consumer agent-skill
library. `omnibias-keras` 0.0.1a1 and `omnibias-qpinn` 0.0.2a1 are the earliest.

Full inventory with maturity tiers: [`docs/packages.md`](docs/packages.md).

### Licensing

Two tiers, and they are not the same licence:

- **Tier P — `Apache-2.0`** (28 packages): the derivative tower and everything
  built directly on it. No copyleft, express patent grant, no commercial
  licence ever required.
- **Tier C — `AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial`**
  (14 packages): the certified-decision layer.

The split is recorded in `[tool.omnibias.license_tiers]` in the root
`pyproject.toml` and enforced by
`packages/omnibias-core/tests/test_license_consistency.py`, which fails the
build if a permissive package ever gains a copyleft dependency. Metadata
follows PEP 639 (`License-Expression`), and the repository follows the REUSE
Specification. See [`LICENSING.md`](LICENSING.md).

### Rigour and honesty

Claims are tiered and the tiers are enforced, not asserted:

- `omnibias.core.verified` produces **sound, outward-rounded enclosures** —
  intervals, affine zonotopes, Taylor models, QR-Lohner validated flow,
  Lehmann–Maehly–Goerisch eigenvalue lower bounds. Every enclosure is tested to
  contain both a dense deterministic grid and a random sample of true values.
- Certificates are canonical, hash-sealed JSON (format v1), tamper-evident via
  `verify_certificate_digest`.
- `theorem_prover_verified` is earned **only** by a genuine `lake build` pass
  against the Mathlib-free Lean kernel in `formal/omnibias-verified-kernel`.
  Asserting it without a pass blocks the verdict; with no Lean toolchain the
  bridge degrades gracefully. `mathlib_verified` is a separate, distinct tier.
- Methods are labelled honestly throughout: closed-form, forward-mode autodiff
  of an analytic quantity, and grid-based approximation are never conflated.
  `omnibias-fractional` is explicitly *not* closed form. Exact submodular
  minimisation is P-class and is not dressed up as more.
- The two senses of "collapse" are kept apart in every source, doc, and skill,
  and a terminology guard fails the build if they are conflated: the founding
  **bias collapse** is the `delta -> 0` limit in which `K` parallel hyperplanes
  coalesce into one carrying the derivative tower `sigma^(K-1)`, while
  **temperature collapse** is the `beta -> inf` limit in which a single
  hyperplane sharpens into a 0/1 feasibility indicator.

### Engineering

- Documentation is **executable**: CI runs every fenced Python block in the
  docs, and opting a block out requires a stated reason.
- Guard tests cover leakage (no absolute paths, scheduler tokens, vendor names,
  or secrets in any readable file), terminology, conceptual lineage, licence
  consistency, packaging hygiene, the public API surface, and agent-skill
  drift. Each guard self-tests its own blocklist so it cannot go vacuous.
- CI: 42 per-package test jobs, cross-backend parity, `ruff`, `mypy --strict`
  on the T1 workspace plus a growing curated-beta allowlist, `mkdocs build
  --strict`, wheel build with `twine check`, clean-venv import smoke, CodeQL,
  OpenSSF Scorecard, dependency review, and SBOM.
- Releases publish through PyPI trusted publishing (OIDC) with SLSA build
  provenance attestations; no long-lived credential exists.

[Unreleased]: https://github.com/derivon-ai/omnibias/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/derivon-ai/omnibias/releases/tag/v0.4.0
