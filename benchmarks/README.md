# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon

# Public benchmarks

Every number the root [`README.md`](../README.md) quotes from this suite is
produced by one of the scripts below and committed as JSON under
[`docs/benchmarks/`](../docs/benchmarks/). Regenerate on any commodity CPU:

```bash
uv sync --all-extras --dev
# folx is optional for the Laplacian / polylaplacian scripts
uv pip install 'folx>=0.2'

uv run python benchmarks/laplacian_scaling.py
uv run python benchmarks/polylaplacian_order.py
uv run python benchmarks/derivative_order.py
uv run python benchmarks/optimizer_pinn.py
uv run python benchmarks/burgers_shock_conservation.py
uv run python benchmarks/hard_conditions_solver.py
```

Then rebuild the figures that plot those JSON files:

```bash
uv run python docs/img/generate_figures.py
```

## What each script measures

| Script | Artifact | Comparison |
|---|---|---|
| `laplacian_scaling.py` | `docs/benchmarks/laplacian_scaling.json` | omnibias / folx / `jax.hessian` / `torch.func.hessian` Laplacian cost vs input dimension `D` |
| `polylaplacian_order.py` | `docs/benchmarks/polylaplacian_order.json` | closed-form `Δᵏ` vs nested folx / nested dense Hessian; records OOM / timeout honestly |
| `derivative_order.py` | `docs/benchmarks/derivative_order.json` | `σ^(n)` closed-form vs nested autograd vs finite differences (cost + accuracy) |
| `optimizer_pinn.py` | `docs/benchmarks/optimizer_pinn.json` | 1-D Poisson PINN: Adam / L-BFGS vs Gauss–Newton / cubic GN / trust-region Newton-CG |
| `burgers_shock_conservation.py` | `docs/benchmarks/burgers_shock_conservation.json` | shock-capturing Burgers PINN, conservative flux-form cage vs non-conservative arm at identical architecture / budget / seed, swept over 6 viscosities x 5 seeds |
| `hard_conditions_solver.py` | `docs/benchmarks/hard_conditions_solver.json` | Poisson / heat / wave with boundary + initial conditions absorbed into the ansatz vs kept as loss terms, at identical architecture / budget / seed over 5 seeds; reports boundary violation and interior relative L2 |
| `information_geometry.py` | `docs/benchmarks/information_geometry_smoke.json` / `information_geometry.json` | Shipped 04-01: Wave-0 falsifier A6 (G2) plus product API G1/G3–G5: two-bias `G_{delta,delta} ~ delta^2/720` and `omnibias.curvature.information` on a randomized mixture suite |
| `conformal_slabs.py` | `docs/benchmarks/conformal_slabs_smoke.json` | Shipped 04-02: three guarantee kinds stay apart; G1–G6 CI; conformal is not sealable |
| `inequality_facade.py` | `docs/benchmarks/inequality_facade_smoke.json` | Shipped 09-30: propose / rationalize / check; G1–G4 CI; G5 leftover-recorded, not in `all_passed`; not a new LP algorithm |
| `convergence_ledger.py` | `docs/benchmarks/convergence_ledger_smoke.json` | Shipped 07-08: NS exponent / YM polymer / NS scale ledgers discharge over `Q`; failing margin is `BLOCKED` and named; parent flags derived, never asserted; `mathlib_verified` stays false on the kernel path |
| `anisotropic_profile.py` | `docs/benchmarks/anisotropic_profile_smoke.json` | Shipped 07-09: Lemma 4.1 operators / axis germ / source jets; honesty keys sealed; not a forced-blowup reproof |
| `stress_cone.py` | `docs/benchmarks/stress_cone_smoke.json` | Shipped 07-10: exact 2x2 cone membership + scale ledger; parallel / opposite cases `BLOCKED` and named |
| `weighted_class.py` | `docs/benchmarks/weighted_class_smoke.json` | Shipped 07-11: fixed-order W/M/S bound + finite Gevrey majorant; not a Gevrey theorem |
| `swirl_heat_pulse.py` | `docs/benchmarks/swirl_heat_pulse_smoke.json` | Shipped 07-12: radial m=1 identity + exact-D envelope; h=0 CI plant; not a 3-D heat theorem |
| `forced_flat_blowup.py` | `docs/benchmarks/forced_flat_blowup_smoke.json` | Shipped 07-13: axis-flat leading stress + bounded core energy; not a Clay (C)/(D) reproof |
| `ns_core_search.py` | `docs/benchmarks/ns_core_search_smoke.json` | Shipped 07-14: finite axis-regular jet family + profile residual + cone reject; not a forced-blowup reproof |
| `ipm_remainder_cap.py` | `docs/benchmarks/ipm_remainder_cap_smoke.json` | Shipped 07-15: CubicGN + leftover #53 toy radii only; not Navier-Stokes |
| `boussinesq_remainder_cap.py` | `docs/benchmarks/boussinesq_remainder_cap_smoke.json` | Shipped 07-16: CubicGN twin + leftover #54; lambda_n stays a hypothesis |
| `pulse_family_composition.py` | `docs/benchmarks/pulse_family_composition_smoke.json` | Shipped 07-17: locked pulse composed into the 07-13 field; scale premises untouched; not a forced-blowup reproof |
| `unforced_bkm_slab.py` | `docs/benchmarks/unforced_bkm_slab_smoke.json` | Shipped 07-18: force-free TG slab + Interval BKM integral; not Clay A/B |
| `unforced_slab_continuation.py` | `docs/benchmarks/unforced_slab_continuation_smoke.json` | Shipped 07-19: continue or Halt; leftover #57; not Clay A/B |
| `force_is_essential.py` | `docs/benchmarks/force_is_essential_smoke.json` | Shipped 07-20: unforced limit of 07-13 is BLOCKED; leftover #58 `f0_not_a_corollary`; not Clay A/B |
| `unforced_abc_slab.py` | `docs/benchmarks/unforced_abc_slab_smoke.json` | Shipped 07-21: 3-D force-free ABC BKM slab + continuation; leftover #57 reused; not Clay A/B |
| `unforced_tg3d_ic.py` | `docs/benchmarks/unforced_tg3d_ic_smoke.json` | Shipped 07-22: 3-D TG is an IC, not closed-form decay; leftover #59; not Clay A/B |
| `unforced_abc_long_chain.py` | `docs/benchmarks/unforced_abc_long_chain_smoke.json` | Shipped 07-23: four ABC slabs cover `[0, 2]`; leftover #57 reused; not Clay A/B |
| `inverse_imaging.py` | `docs/benchmarks/inverse_imaging_smoke.json` / `inverse_imaging.json` | Shipped 05-01: Wave-0 falsifier A7 (G7) plus product API G1–G6: locally-seeded `sd(tau_hat) ~ alpha^(n - 5/2)`; `omnibias.pinn.inverse` localization / layered / Stefan / sensors |
| `tabular_arrangement.py` | `docs/benchmarks/tabular_arrangement_smoke.json` / `tabular_arrangement.json` | Shipped 05-02 G1/G2: H=2 arrangement vs tuned LightGBM on constructed oblique XOR / axis AND; fair early-stop protocol (train Xtr, stop Xva, score Xte, no train+val refit) |
| `tabular_arrangement_public.py` | `docs/benchmarks/tabular_arrangement_public_smoke.json` / `tabular_arrangement_public.json` | Shipped 05-02 G3: same fair protocol on eight public binary datasets (full win/loss table; G4 leftover-recorded leftover #50 from eight-dataset artifact, predictiveness `0.25`, not in `all_passed`) |
| `tabular_arrangement_capacity.py` | `docs/benchmarks/tabular_arrangement_capacity_smoke.json` / `tabular_arrangement_capacity.json` | Shipped 05-02 G3b leftover-recorded leftover #49: capacity/optimizer ablations; G3 frozen; `boost_h2` not-worse `4/8` (need `>=6/8`); smoke 1-dataset score is not G3b; not in `all_passed` |
| `tabular_pou.py` | `docs/benchmarks/tabular_pou_smoke.json` | Designed 05-04: axis POU booster G0–G6; G3 boost-only v0 freeze for 05-05; smoke G0/G5; RealMLP/TabM `--full` only |
| `tabular_pou_joint.py` | `docs/benchmarks/tabular_pou_joint_smoke.json` / `tabular_pou_joint.json` | Shipped 05-05: frozen sequential residual (H1); G-tree `3/3` / G-net `2/3` / G-hybrid `6/6` / G5 earned; H0/H3/H4/H6 rejected; smoke H0 identity + tiny H1 + G5; temperature collapse only |
| `shape_topology.py` | `docs/benchmarks/shape_topology_smoke.json` / `$OMNIBIAS_SCRATCH/beyond_pde/shape_topology.json` | Shipped 05-02 G6/G7: soft Euler gap containment + topology-regularized genus vs a named soft-disk implicit; G4 unearned; G5 failed |
| `sequence_transverse.py` | `docs/benchmarks/sequence_transverse_smoke.json` / `$OMNIBIAS_SCRATCH/beyond_pde/sequence_transverse.json` | Shipped 05-02 G5 / Wave-0 A5: order-0 causal `sigma` FIR vs named S4D (N=1) on AR(1); `width=T`; matched 4 params; earned |
| `jet_vs_nested_ad.py` | `docs/benchmarks/jet_vs_nested_ad_smoke.json` / `$OMNIBIAS_SCRATCH/citation/jet_vs_ad/` | Shipped 06-05 obligation 3: 1-D Poisson `mlp_jet` vs nested AD + GN vs Adam; not CCF; extract / paper stay later |
| `multipack_birkhoff.py` | `docs/benchmarks/multipack_birkhoff_smoke.json` | Wave-1 primitive 01-01: MultiPackUnit G1–G5; float64 order ceiling recorded; two-interface span beats OperatorBlock / OMBU / JetMLP |
| `irregular_stencils.py` | `docs/benchmarks/irregular_stencils_smoke.json` | Shipped 01-04: exact-Q Birkhoff weights G1–G4 |
| `bias_scan.py` | `docs/benchmarks/bias_scan_smoke.json` | Wave-1 primitive 01-02: BiasScan G1–G4 CI-gated; 01-13 G5, **shipped**; G4 is a warmed-up voxelize-then-`cmbConv1d` pipeline comparison |
| `mollifier_calculus.py` | `docs/benchmarks/mollifier_calculus_smoke.json` | Shipped 01-05: MollifierSpec G1–G4 CI-gated; certified exponential tails, not compact support; G4 is exact vs matched-cost Gauss on a known Poisson |
| `spectral_design.py` | `docs/benchmarks/spectral_design_smoke.json` | Shipped 01-07: BandPlan G1/G2/G4; G3 leftover-recorded (`0/5` Mscale hits of the four-gap lstsq gate), not in `all_passed`; pack order is a band selector |
| `arrangement_geometry.py` | `docs/benchmarks/arrangement_geometry_smoke.json` | Shipped 01-03: G1–G4 CI; temperature collapse; sampled subgraph; cost vs n/D leftover-recorded, not in CI `all_passed` |
| `ombu_frames.py` | `docs/benchmarks/ombu_frames_smoke.json` | Shipped 01-06: `sigma'` not admissible; G4 denoising leftover-recorded (`5/5` MSE vs n=1, `0/5` skill vs identity), not in `all_passed` |
| `tropical_homotopy.py` | `docs/benchmarks/tropical_homotopy_smoke.json` | Shipped 01-08: reuses `logsumexp_gap_bound`; G4 path-following earned; cost vs n/D leftover-recorded |
| `equality_locus.py` | `docs/benchmarks/equality_locus_smoke.json` | Shipped 01-09: constraint manifold, not a PDE solver |
| `jet_bundle.py` | `docs/benchmarks/jet_bundle_smoke.json` | Shipped 01-10: vocabulary, not a discovery; G1–G3 earned |
| `enclosure_collapse.py` | `docs/benchmarks/enclosure_collapse_smoke.json` | Shipped 01-14: Width Law + six squeezes; `width -> 0` of a sound enclosure (a point plus a proof); not a package |
| `jet_width_vs_order.py` | `docs/benchmarks/jet_width_vs_order_smoke.json` / `$OMNIBIAS_SCRATCH/jet_width_vs_order/jet_width_vs_order.json` | Certified 1-D `certified_partials` width sweep; grid + random coverage; raw factorial envelope and factorial-normalized width exponent; finite MLP evidence, not a spectral-tail theorem |
| `dirichlet_enclosure.py` | `docs/benchmarks/dirichlet_enclosure_smoke.json` | `Re(s)>1` width + coverage; no zeros / no RH subject |
| `certified_continuation.py` | `docs/benchmarks/certified_continuation_smoke.json` | Finite FE / AFE / winding / `H_t` pack; not RH |
| `instance_gap_tightening.py` | `docs/benchmarks/instance_gap_tightening_smoke.json` | Named n<=8 Lasserre 1 vs 2; never tight; never P = NP |
| `ccf_pade_profile.py` | `docs/benchmarks/ccf_pade_profile_smoke.json` | Jet-Padé profile diagnostic; stretch 1e-13 untouched |
| `theory_homes.py` | `docs/benchmarks/theory_homes_smoke.json` | Shipped 06-03: G1–G5 earned (125/125 homes; Wave-0 A4–A7 recorded; G4/G5 vacuous; 389 lines, not promoted) |
| `conjugate_hilbert.py` | `docs/benchmarks/conjugate_hilbert_smoke.json` | Shipped 01-12: line Hilbert; G1–G4 CI; G5 leftover-recorded (leftover #11; matched-width ratio `0.978`, need `10x`), not in `all_passed` |
| `arrangement_graph.py` | `docs/benchmarks/arrangement_graph_smoke.json` | Shipped 02-02 Face-Net: sampled subgraph; G3 vs k-NN leftover-recorded (0-hop; GNN / RegionModels `--full`); cost vs n/D leftover-recorded |
| `bem_net.py` | `docs/benchmarks/bem_net_smoke.json` | Shipped 02-06: off-surface exact; G2 disc earned (`circle_dirichlet_density`); G4 `eps^2` regularization order earned; G3 exterior win leftover-recorded (no volume PINN); single-layer cost leftover-recorded (leftover #42) |
| `pack_tree.py` | `docs/benchmarks/pack_tree_smoke.json` | Shipped 02-07: 1-D offsets; G1–G5 earned (G3 leftover #13 closed; G4 `_deriv_bound` separation) |
| `equivariant_scan.py` | `docs/benchmarks/equivariant_scan_smoke.json` | Shipped 02-08: gaussian steering; discrete `C_L`; G1–G4 earned; G5 leftover-recorded (`--full`); orbit cost leftover-recorded (leftover #43) |
| `soliton_tanh_method.py` | `docs/benchmarks/soliton_tanh_method_smoke.json` | Shipped 02-09: tanh algebra; G1/G2/G3/G5 earned; G4 init-win leftover-recorded (`--full`); algebraic cost leftover-recorded (leftover #44) |
| `hermite_ladder.py` | `docs/benchmarks/hermite_ladder_smoke.json` | Shipped 02-10: Rodrigues reweight; G1–G3/G6 earned; G4 many-body leftover-recorded (`--full`); exact-vs-FD leftover-recorded (leftover #45); G5 anharmonic leftover-recorded |
| `layered_transfer.py` | `docs/benchmarks/layered_transfer_smoke.json` | Shipped 02-11: `continuum_claim=False`; G1–G3/G6 earned; G4 inverse-design leftover-recorded (`--full`); stack cost leftover-recorded (leftover #46); G5 conservation leftover-recorded |
| `equality_intersection.py` | `docs/benchmarks/equality_intersection_smoke.json` | Shipped 02-12: not a general PDE solver; G1–G3/G5/G6 earned; G4 Burgers RH leftover-recorded (noisy contour `--full`) |
| `linearizing_transforms.py` | `docs/benchmarks/linearizing_transforms_smoke.json` | Shipped 02-13: named transforms; G1/G6 CI; G2 leftover-recorded (n-soliton); G3 leftover-recorded (no train vs direct PINN); G4 leftover-recorded (permutability); 03-11 stays designed |
| `holonomy_band.py` | `docs/benchmarks/holonomy_band_smoke.json` | Shipped 02-14: no YM / mass-gap claim; G2 earned; G3 leftover-recorded; G4 earned (`random_u1_gauge`) |
| `gauge_holonomy_gap.py` | `docs/benchmarks/gauge_holonomy_gap_smoke.json` | Shipped 07-04: holonomy trials on one fixed matrix; G1 factor measured; no YM / continuum |
| `gauge_two_plaquette_gap.py` | `docs/benchmarks/gauge_two_plaquette_gap_smoke.json` | Shipped 07-04: two-plaquette KS Hamiltonian `λ1-λ0`; G1 factor measured; no YM / continuum |
| `gauge_spatial_strip.py` | `docs/benchmarks/gauge_spatial_strip_smoke.json` | Shipped 07-04: finite 2+1-D strip gap + RP + cluster tail; no YM / OS / continuum |
| `gauge_gap_scaling.py` | `docs/benchmarks/gauge_gap_scaling_smoke.json` | Shipped 07-04: independent certified gaps vs spacing; continuum_claim false |
| `gauge_finite_report.py` | `docs/benchmarks/gauge_finite_report_smoke.json` | Shipped 07-04: sealed finite-gauge pack; SU(3) gap at locked `n_cells=32`; three-plaquette G1 measured; Wilson-character domain; no YM / continuum / Clay staircase |
| `scannet.py` | `docs/benchmarks/scannet_smoke.json` | Shipped 02-01: ScanNet G1/G2/G3/G5; G3 wall/point vs named k-NN over two decades is in CI `all_passed`; G4 leftover-recorded (leftover #17); on-lattice equivariance |
| `jetkan.py` | `docs/benchmarks/jetkan_smoke.json` | Shipped 02-03: JetKAN G1/G3/G4/G5; G2 leftover-recorded (leftover #41; ~2.2x vs autodiff, need 5x); model-jet exactness, KA theorem does not justify |
| `weak_form_vpinn.py` | `docs/benchmarks/weak_form_vpinn_smoke.json` | Shipped 02-04: exact on polynomial boxes; G4 conditioning earned vs strong collocation |
| `multi_interface_pinn.py` | `docs/benchmarks/multi_interface_pinn_smoke.json` | Shipped 02-05: sharpening, neither collapse; G3 leftover-recorded (linear stand-in); G4 leftover-recorded (zero-coeff; training `--full`) |
| `jet_line_search.py` | `docs/benchmarks/jet_line_search_smoke.json` | Shipped 03-12: G1/G2/G3/G6 CI-gated; G4 leftover-recorded vs strong Wolfe (leftover #47; `1.83x`, need `2x`); G5 leftover-recorded (leftover #48); not in CI `all_passed` |
| `adaptive_refinement.py` | `docs/benchmarks/adaptive_refinement_smoke.json` | Shipped 03-13: G1–G6 CI-gated; G4 10x vs matched-count fixed on the named BL is in CI `all_passed` |

All runs are **float64**, **CPU** (`JAX_PLATFORMS=cpu`). Each JSON carries
`generated_utc`, `hardware_class`, library versions, and the exact config.

## Theory-program falsifiers

Wave-0 kill experiments from [`theory/06-program/03-packaging-and-rollout.md`](../theory/06-program/03-packaging-and-rollout.md).
Shared gate protocol (Shipped 06-01): [`theory/06-program/01-acceptance-gates-and-benchmarks.md`](../theory/06-program/01-acceptance-gates-and-benchmarks.md).
Helpers: `require_scaling_exponent`, `require_rel_error`, `require_within_stderr`,
`require_capture_rate`, `require_all_seeds`, `require_enclosure_coverage`,
`require_backend_parity`, `require_cost_parity` in [`_gates.py`](_gates.py),
self-tested in `tests/test_gates_protocol.py`. Artifact classes and G5
(`baseline.name`) live in [`_schema.py`](_schema.py).

```bash
# CI smoke
uv run python benchmarks/information_geometry.py
uv run python benchmarks/inverse_imaging.py
uv run python benchmarks/tabular_arrangement.py
uv run python benchmarks/tabular_arrangement_public.py
uv run python benchmarks/tabular_arrangement_capacity.py
uv run python benchmarks/shape_topology.py
uv run python benchmarks/jet_vs_nested_ad.py
uv run python benchmarks/multipack_birkhoff.py
uv run python benchmarks/irregular_stencils.py
uv run python benchmarks/bias_scan.py
# Multi-seed / multi-realization acceptance
uv run python benchmarks/information_geometry.py --full
uv run python benchmarks/inverse_imaging.py --full
uv run python benchmarks/tabular_arrangement.py --full
uv run python benchmarks/tabular_arrangement_public.py --full
uv run python benchmarks/tabular_arrangement_capacity.py --full
uv run python benchmarks/shape_topology.py --full
uv run python benchmarks/jet_vs_nested_ad.py --full
uv run python benchmarks/multipack_birkhoff.py --full
uv run python benchmarks/irregular_stencils.py --full
uv run python benchmarks/bias_scan.py --full
uv run python benchmarks/mollifier_calculus.py --full
uv run python benchmarks/spectral_design.py --full
uv run python benchmarks/scannet.py --full
uv run python benchmarks/jetkan.py --full
uv run python benchmarks/weak_form_vpinn.py --full
uv run python benchmarks/multi_interface_pinn.py --full
```

## PINN four-gap suite

Acceptance-gated scripts that close the four named PINN capability gaps.
Capability matrix:
[`docs/benchmarks/pinn_four_gap_matrix.md`](../docs/benchmarks/pinn_four_gap_matrix.md).

| Script | Gap | Smoke artifact | Full (`--full`) artifact |
|---|---|---|---|
| `causal_marching.py` | Causality (`omnibias.pinn.train`) | `docs/benchmarks/causal_marching_smoke.json` | `docs/benchmarks/causal_marching.json` |
| `geometry_sdf.py` | Geometry (`omnibias.pinn.domain`) | `docs/benchmarks/geometry_sdf_smoke.json` | `docs/benchmarks/geometry_sdf.json` |
| `operator_zero_shot.py` | Operators (`omnibias.pinn.operator`) | `docs/benchmarks/operator_zero_shot_smoke.json` | `docs/benchmarks/operator_zero_shot.json` |
| `spectral_bias_fbpinn.py` | Spectral bias (FBPINN + one-shot `lstsq`) | `docs/benchmarks/spectral_bias_fbpinn_smoke.json` | `docs/benchmarks/spectral_bias_fbpinn.json` |

### Smoke vs `--full`

| Mode | How | Seeds / budget | Role |
|---|---|---|---|
| **Smoke** (default) | `uv run python benchmarks/<script>.py` | 1 seed, tiny nets / steps | CI wiring gate; writes `*_smoke.json` under `docs/benchmarks/` |
| **Full** | `uv run python benchmarks/<script>.py --full` | multi-seed acceptance (typically 5) | Commit the summary JSON under `docs/benchmarks/`; heavier copies may also land under `$OMNIBIAS_SCRATCH` (default `artifacts/`) |

Never substitute a smoke JSON for a multi-seed acceptance claim. CI runs the
four scripts in smoke mode on every push.

### Absolute gates (`benchmarks/_gates.py`)

Shared helpers live in [`_gates.py`](_gates.py). Every four-gap artifact emits a
top-level `gates` block that self-declares pass/fail so a JSON can never look
like a result while encoding a divergence. Gates answer three questions in
order:

1. Is the **reference** physically valid? (maximum principle, known amplitude, …)
2. Does every prediction beat the **zero predictor**? (`skill_score > 0`)
3. Does absolute error clear a **named threshold**? (`rel_l2 <= tol`, or a
   conditioned-vs-baseline comparison)

Helpers: `rel_l2`, `skill_score`, `require_reference_valid`, `gates_block`,
`require_scaling_exponent`, `require_rel_error`, `require_within_stderr`,
`require_capture_rate`.

### Regenerate the four-gap artifacts

```bash
# CI smoke (writes *_smoke.json)
uv run python benchmarks/causal_marching.py
uv run python benchmarks/geometry_sdf.py
uv run python benchmarks/operator_zero_shot.py
uv run python benchmarks/spectral_bias_fbpinn.py

# Multi-seed acceptance (writes the committed full JSON + optional scratch copy)
uv run python benchmarks/causal_marching.py --full
uv run python benchmarks/geometry_sdf.py --full
uv run python benchmarks/operator_zero_shot.py --full
uv run python benchmarks/spectral_bias_fbpinn.py --full
```

## Hardware tiers

| Tier | Where numbers live | Reproducible from this repo? |
|---|---|---|
| **CPU (this suite)** | `docs/benchmarks/*.json` | **Yes** — run the scripts above |
| **Data-center GPU (off-band)** | tables in [`docs/complexity.md`](../docs/complexity.md) labelled *off-band* | Measured on a separate GPU host; scripts not required for the CPU claims |

The README keeps these tiers visually separate. Do not paste an off-band GPU
number into a "reproduce this" block.

## Dependencies

- Required: `omnibias-core`, `omnibias-torch`, `omnibias-jax`, `numpy`, `torch`, `jax`.
- Optional: `folx` (Laplacian and polylaplacian scripts). Without it those two
  scripts exit with an import error — install it to regenerate those artifacts.
  Four-gap scripts additionally need `omnibias-pinn` (workspace install).
