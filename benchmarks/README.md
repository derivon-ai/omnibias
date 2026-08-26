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
| `information_geometry.py` | `docs/benchmarks/information_geometry_smoke.json` / `information_geometry.json` | Wave-0 falsifier A6 (04-01 G2) plus product API G1/G3–G5: two-bias `G_{delta,delta} ~ delta^2/720` and `omnibias.curvature.information` on a randomized mixture suite |
| `inverse_imaging.py` | `docs/benchmarks/inverse_imaging_smoke.json` / `inverse_imaging.json` | Wave-0 falsifier A7 (05-01 G7) plus product API G1–G6: locally-seeded `sd(tau_hat) ~ alpha^(n - 5/2)`; `omnibias.pinn.inverse` localization / layered / Stefan / sensors |
| `tabular_arrangement.py` | `docs/benchmarks/tabular_arrangement_smoke.json` / `tabular_arrangement.json` | Wave-0 falsifier A4 (05-02 G1/G2): H=2 arrangement vs tuned LightGBM on constructed oblique XOR / axis AND; fair early-stop protocol (train Xtr, stop Xva, score Xte, no train+val refit) |
| `tabular_arrangement_public.py` | `docs/benchmarks/tabular_arrangement_public_smoke.json` / `tabular_arrangement_public.json` | 05-02 G3: same fair protocol on eight public binary datasets (full win/loss table; G4 reported from eight-dataset artifact, predictiveness `0.25`, not in `all_passed`) |
| `tabular_arrangement_capacity.py` | `docs/benchmarks/tabular_arrangement_capacity_smoke.json` / `tabular_arrangement_capacity.json` | 05-02 G3b: capacity/optimizer ablations; G3 frozen; `boost_h2` not-worse `4/8` (need `>=6/8`); smoke 1-dataset score is not G3b; not in `all_passed` |
| `shape_topology.py` | `docs/benchmarks/shape_topology_smoke.json` / `$OMNIBIAS_SCRATCH/beyond_pde/shape_topology.json` | 05-02 G6/G7: soft Euler gap containment + topology-regularized genus vs a named soft-disk implicit; G4 unearned; G5 failed |
| `sequence_transverse.py` | `docs/benchmarks/sequence_transverse_smoke.json` / `$OMNIBIAS_SCRATCH/beyond_pde/sequence_transverse.json` | Wave-0 A5 (05-02 G5): order-0 causal `sigma` FIR vs named S4D (N=1) on AR(1); `width=T`; matched 4 params; earned |
| `jet_vs_nested_ad.py` | `docs/benchmarks/jet_vs_nested_ad_smoke.json` / `$OMNIBIAS_SCRATCH/citation/jet_vs_ad/` | 06-05 obligation 3: 1-D Poisson `mlp_jet` vs nested AD + GN vs Adam; not CCF; extract / paper stay later |
| `multipack_birkhoff.py` | `docs/benchmarks/multipack_birkhoff_smoke.json` | Wave-1 primitive 01-01: MultiPackUnit G1–G5; float64 order ceiling recorded; two-interface span beats OperatorBlock / OMBU / JetMLP |
| `irregular_stencils.py` | `docs/benchmarks/irregular_stencils_smoke.json` | Shipped 01-04: exact-Q Birkhoff weights G1–G4 |
| `bias_scan.py` | `docs/benchmarks/bias_scan_smoke.json` | Wave-1 primitive 01-02: BiasScan G1–G4 CI-gated; 01-13 G5; G4 is a warmed-up voxelize-then-`cmbConv1d` pipeline comparison |
| `mollifier_calculus.py` | `docs/benchmarks/mollifier_calculus_smoke.json` | Shipped 01-05: MollifierSpec G1–G4 CI-gated; certified exponential tails, not compact support; G4 is exact vs matched-cost Gauss on a known Poisson |
| `spectral_design.py` | `docs/benchmarks/spectral_design_smoke.json` | Wave-3 primitive 01-07: BandPlan G1/G2/G4; G3 reported (`0/5` Mscale hits of the four-gap lstsq gate), not in `all_passed`; pack order is a band selector |
| `arrangement_geometry.py` | `docs/benchmarks/arrangement_geometry_smoke.json` | Shipped 01-03: temperature collapse; sampled subgraph; cost vs n/D leftover-recorded, not in CI `all_passed` |
| `ombu_frames.py` | `docs/benchmarks/ombu_frames_smoke.json` | Shipped 01-06: `sigma'` not admissible; G4 denoising leftover-recorded (`5/5` MSE vs n=1, `0/5` skill vs identity), not in `all_passed` |
| `tropical_homotopy.py` | `docs/benchmarks/tropical_homotopy_smoke.json` | Gated 01-08: reuses `logsumexp_gap_bound`; G4 path-following earned; cost vs n/D leftover-recorded |
| `equality_locus.py` | `docs/benchmarks/equality_locus_smoke.json` | Gated 01-09: constraint manifold, not a PDE solver |
| `jet_bundle.py` | `docs/benchmarks/jet_bundle_smoke.json` | Gated 01-10: vocabulary, not a discovery; G1–G3 earned |
| `enclosure_collapse.py` | `docs/benchmarks/enclosure_collapse_smoke.json` | Gated 01-14: Width Law + six squeezes; `width -> 0` of a sound enclosure (a point plus a proof); not a package |
| `theory_homes.py` | `docs/benchmarks/theory_homes_smoke.json` | Gated 06-03: G1–G5 earned (94/94 homes; Wave-0 A4–A7 recorded; G4/G5 vacuous; 389 lines, not promoted) |
| `conjugate_hilbert.py` | `docs/benchmarks/conjugate_hilbert_smoke.json` | Gated 01-12: line Hilbert; G1–G4 CI; G5 reported on CCF smoke (matched-width ratio `0.978`, need `10x`), not in `all_passed` |
| `arrangement_graph.py` | `docs/benchmarks/arrangement_graph_smoke.json` | Gated 02-02 Face-Net: sampled subgraph; G3 vs k-NN leftover-recorded (0-hop; GNN / RegionModels `--full`); cost vs n/D leftover-recorded |
| `bem_net.py` | `docs/benchmarks/bem_net_smoke.json` | Gated 02-06: off-surface exact; G2 disc earned (`circle_dirichlet_density`); G3 exterior win leftover-recorded (no volume PINN); single-layer cost reported |
| `pack_tree.py` | `docs/benchmarks/pack_tree_smoke.json` | Gated 02-07: 1-D offsets; G3 earned (cached O(p) multipole, dense crossover) |
| `equivariant_scan.py` | `docs/benchmarks/equivariant_scan_smoke.json` | Gated 02-08: gaussian steering; discrete `C_L`; G5 leftover-recorded (`--full`); orbit cost reported |
| `soliton_tanh_method.py` | `docs/benchmarks/soliton_tanh_method_smoke.json` | Gated 02-09: tanh algebra; G4 init-win leftover-recorded (`--full`); algebraic cost reported |
| `hermite_ladder.py` | `docs/benchmarks/hermite_ladder_smoke.json` | Gated 02-10: Rodrigues reweight; G4 many-body leftover-recorded (`--full`); exact-vs-FD reported; G5 anharmonic leftover-recorded |
| `layered_transfer.py` | `docs/benchmarks/layered_transfer_smoke.json` | Gated 02-11: `continuum_claim=False`; G4 inverse-design leftover-recorded (`--full`); stack cost reported; G5 conservation leftover-recorded |
| `equality_intersection.py` | `docs/benchmarks/equality_intersection_smoke.json` | Gated 02-12: not a general PDE solver; G4 Burgers RH leftover-recorded (noisy contour `--full`) |
| `linearizing_transforms.py` | `docs/benchmarks/linearizing_transforms_smoke.json` | Gated 02-13: named transforms; G1 jet identity CI; G3 leftover-recorded (no train vs direct PINN); 03-11 stays designed |
| `holonomy_band.py` | `docs/benchmarks/holonomy_band_smoke.json` | Gated 02-14: no YM / mass-gap claim; G2 earned; G3 leftover-recorded; G4 earned (`random_u1_gauge`) |
| `gauge_holonomy_gap.py` | `docs/benchmarks/gauge_holonomy_gap_smoke.json` | Gated 07-04: holonomy trials on one fixed matrix; G1 factor measured; no YM / continuum |
| `gauge_two_plaquette_gap.py` | `docs/benchmarks/gauge_two_plaquette_gap_smoke.json` | Two-plaquette KS Hamiltonian `λ1-λ0`; G1 factor measured; no YM / continuum |
| `gauge_spatial_strip.py` | `docs/benchmarks/gauge_spatial_strip_smoke.json` | Finite 2+1-D strip gap + RP + cluster tail; no YM / OS / continuum |
| `gauge_gap_scaling.py` | `docs/benchmarks/gauge_gap_scaling_smoke.json` | Independent certified gaps vs spacing; continuum_claim false |
| `gauge_finite_report.py` | `docs/benchmarks/gauge_finite_report_smoke.json` | Sealed finite-gauge pack; SU(3) gap at locked `n_cells=32`; three-plaquette G1 measured; Wilson-character domain; no YM / continuum / Clay staircase |
| `scannet.py` | `docs/benchmarks/scannet_smoke.json` | Wave-3 architecture 02-01: ScanNet G1/G2/G3/G5; G3 wall/point vs named k-NN over two decades is in CI `all_passed`; G4 density boundary reported (Scan-Net wins the constructive fit); on-lattice equivariance |
| `jetkan.py` | `docs/benchmarks/jetkan_smoke.json` | Wave-3 architecture 02-03: JetKAN G1/G3/G5; G2 cost reported (~2.2x vs autodiff, need 5x); model-jet exactness, KA theorem does not justify |
| `weak_form_vpinn.py` | `docs/benchmarks/weak_form_vpinn_smoke.json` | Wave-3 architecture 02-04: exact on polynomial boxes; G4 conditioning earned vs strong collocation |
| `multi_interface_pinn.py` | `docs/benchmarks/multi_interface_pinn_smoke.json` | Wave-3 architecture 02-05: sharpening, neither collapse; G3 leftover-recorded (linear stand-in); G4 leftover-recorded (zero-coeff; training `--full`) |
| `jet_line_search.py` | `docs/benchmarks/jet_line_search_smoke.json` | Wave-3 algorithm 03-12: G1/G2/G3/G6 CI-gated; G4 reported vs strong Wolfe (`1.83x`, need `2x`); G5 order×depth crossover reported; not in CI `all_passed` |
| `adaptive_refinement.py` | `docs/benchmarks/adaptive_refinement_smoke.json` | Wave-3 algorithm 03-13: G1–G6 CI-gated; G4 10x vs matched-count fixed on the named BL is in CI `all_passed` |

All runs are **float64**, **CPU** (`JAX_PLATFORMS=cpu`). Each JSON carries
`generated_utc`, `hardware_class`, library versions, and the exact config.

## Theory-program falsifiers

Wave-0 kill experiments from [`theory/06-program/03-packaging-and-rollout.md`](../theory/06-program/03-packaging-and-rollout.md).
Shared gate protocol: [`theory/06-program/01-acceptance-gates-and-benchmarks.md`](../theory/06-program/01-acceptance-gates-and-benchmarks.md).
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
