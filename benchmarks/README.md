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
| `information_geometry.py` | `docs/benchmarks/information_geometry_smoke.json` / `information_geometry.json` | Wave-0 falsifier A6 (04-01 G2): Fisher `G_{delta,delta}` exponent `2.00 +- 0.02` and prefactor `1/720` for the two-bias logistic pack vs Monte Carlo Fisher |
| `inverse_imaging.py` | `docs/benchmarks/inverse_imaging_smoke.json` / `inverse_imaging.json` | Wave-0 falsifier A7 (05-01 G7): locally-seeded scan localization `sd(tau_hat) ~ alpha^(n - 5/2)` for `n in {3, 4}` over 5 seeds; global search earned for n=3 only; G1–G6 unearned |
| `tabular_arrangement.py` | `docs/benchmarks/tabular_arrangement_smoke.json` / `tabular_arrangement.json` | Wave-0 falsifier A4 (05-02 G1/G2): H=2 arrangement vs tuned LightGBM on constructed oblique XOR / axis AND; fair early-stop protocol (train Xtr, stop Xva, score Xte, no train+val refit) |
| `tabular_arrangement_public.py` | `docs/benchmarks/tabular_arrangement_public_smoke.json` / `tabular_arrangement_public.json` | 05-02 G3: same fair protocol on eight public binary datasets (full win/loss table; G4 reported, not retuned) |
| `tabular_arrangement_capacity.py` | `docs/benchmarks/tabular_arrangement_capacity_smoke.json` / `tabular_arrangement_capacity.json` | 05-02 G3b: capacity/optimizer ablations; G3 frozen; primary `boost_h2` not-worse on 4/8 (need >=6/8; unearned) |
| `multipack_birkhoff.py` | `docs/benchmarks/multipack_birkhoff_smoke.json` | Wave-1 primitive 01-01: MultiPackUnit G1/G2/G3/G5; float64 order ceiling recorded; G4 deferred |
| `irregular_stencils.py` | `docs/benchmarks/irregular_stencils_smoke.json` | Wave-1 primitive 01-04: exact-Q Birkhoff weights G1–G4 |
| `bias_scan.py` | `docs/benchmarks/bias_scan_smoke.json` | Wave-1 primitive 01-02: BiasScan G1/G2/G3 CI-gated; G4 earned on smoke (not in CI `all_passed`; wall time) |
| `mollifier_calculus.py` | `docs/benchmarks/mollifier_calculus_smoke.json` | Wave-3 primitive 01-05: MollifierSpec G1–G3; G4 deferred to VPINN; certified exponential tails, not compact support |
| `spectral_design.py` | `docs/benchmarks/spectral_design_smoke.json` | Wave-3 primitive 01-07: BandPlan G1/G2/G4; G3 not in CI `all_passed`; pack order is a band selector |
| `arrangement_geometry.py` | `docs/benchmarks/arrangement_geometry_smoke.json` | Gated 01-03: temperature collapse; sampled subgraph |
| `ombu_frames.py` | `docs/benchmarks/ombu_frames_smoke.json` | Gated 01-06: `sigma'` not admissible; G4 not in CI `all_passed` |
| `tropical_homotopy.py` | `docs/benchmarks/tropical_homotopy_smoke.json` | Gated 01-08: reuses `logsumexp_gap_bound`; G4 `--full` |
| `equality_locus.py` | `docs/benchmarks/equality_locus_smoke.json` | Gated 01-09: constraint manifold, not a PDE solver |
| `conjugate_hilbert.py` | `docs/benchmarks/conjugate_hilbert_smoke.json` | Gated 01-12: line Hilbert; G5 not in CI `all_passed` |
| `arrangement_graph.py` | `docs/benchmarks/arrangement_graph_smoke.json` | Gated 02-02 Face-Net: sampled subgraph |
| `bem_net.py` | `docs/benchmarks/bem_net_smoke.json` | Gated 02-06: off-surface exact; BC approximated |
| `pack_tree.py` | `docs/benchmarks/pack_tree_smoke.json` | Gated 02-07: 1-D offsets; G3 cost not in CI `all_passed` |
| `equivariant_scan.py` | `docs/benchmarks/equivariant_scan_smoke.json` | Gated 02-08: gaussian steering; discrete `C_L` |
| `soliton_tanh_method.py` | `docs/benchmarks/soliton_tanh_method_smoke.json` | Gated 02-09: tanh algebra, not a collapse |
| `hermite_ladder.py` | `docs/benchmarks/hermite_ladder_smoke.json` | Gated 02-10: Rodrigues reweight; G4 `--full` |
| `layered_transfer.py` | `docs/benchmarks/layered_transfer_smoke.json` | Gated 02-11: `continuum_claim=False` |
| `equality_intersection.py` | `docs/benchmarks/equality_intersection_smoke.json` | Gated 02-12: not a general PDE solver |
| `linearizing_transforms.py` | `docs/benchmarks/linearizing_transforms_smoke.json` | Gated 02-13: named transforms; 03-11 stays designed |
| `holonomy_band.py` | `docs/benchmarks/holonomy_band_smoke.json` | Gated 02-14: no YM / mass-gap claim |
| `gauge_holonomy_gap.py` | `docs/benchmarks/gauge_holonomy_gap_smoke.json` | Gated 07-04: holonomy trials on one fixed matrix; G1 factor measured; no YM / continuum |
| `gauge_two_plaquette_gap.py` | `docs/benchmarks/gauge_two_plaquette_gap_smoke.json` | Two-plaquette KS Hamiltonian `λ1-λ0`; G1 factor measured; no YM / continuum |
| `gauge_spatial_strip.py` | `docs/benchmarks/gauge_spatial_strip_smoke.json` | Finite 2+1-D strip gap + RP + cluster tail; no YM / OS / continuum |
| `gauge_gap_scaling.py` | `docs/benchmarks/gauge_gap_scaling_smoke.json` | Independent certified gaps vs spacing; continuum_claim false |
| `gauge_finite_report.py` | `docs/benchmarks/gauge_finite_report_smoke.json` | Sealed finite-gauge pack; cellwise SU(3) gap; Wilson-character domain; G1 measured; no YM / continuum / Clay staircase |
| `scannet.py` | `docs/benchmarks/scannet_smoke.json` | Wave-3 architecture 02-01: ScanNet G1/G2/G5; G3 cost and G4 k-NN recorded, not in CI `all_passed`; on-lattice equivariance |
| `jetkan.py` | `docs/benchmarks/jetkan_smoke.json` | Wave-3 architecture 02-03: JetKAN G1/G3/G5; G2 timing not in CI `all_passed`; model-jet exactness, KA theorem does not justify |
| `weak_form_vpinn.py` | `docs/benchmarks/weak_form_vpinn_smoke.json` | Wave-3 architecture 02-04: exact on polynomial boxes; boundary bound on by default |
| `multi_interface_pinn.py` | `docs/benchmarks/multi_interface_pinn_smoke.json` | Wave-3 architecture 02-05: parallel interfaces; `alpha -> inf` is sharpening, neither collapse |

All runs are **float64**, **CPU** (`JAX_PLATFORMS=cpu`). Each JSON carries
`generated_utc`, `hardware_class`, library versions, and the exact config.

## Theory-program falsifiers

Wave-0 kill experiments from [`theory/06-program/03-packaging-and-rollout.md`](../theory/06-program/03-packaging-and-rollout.md).
Shared gate protocol: [`theory/06-program/01-acceptance-gates-and-benchmarks.md`](../theory/06-program/01-acceptance-gates-and-benchmarks.md).
Helpers: `require_scaling_exponent`, `require_rel_error`, `require_within_stderr`,
`require_capture_rate`, `require_all_seeds` in [`_gates.py`](_gates.py), self-tested in
`tests/test_gates_protocol.py`.

```bash
# CI smoke
uv run python benchmarks/information_geometry.py
uv run python benchmarks/inverse_imaging.py
uv run python benchmarks/tabular_arrangement.py
uv run python benchmarks/tabular_arrangement_public.py
uv run python benchmarks/tabular_arrangement_capacity.py
uv run python benchmarks/multipack_birkhoff.py
uv run python benchmarks/irregular_stencils.py
uv run python benchmarks/bias_scan.py
# Multi-seed / multi-realization acceptance
uv run python benchmarks/information_geometry.py --full
uv run python benchmarks/inverse_imaging.py --full
uv run python benchmarks/tabular_arrangement.py --full
uv run python benchmarks/tabular_arrangement_public.py --full
uv run python benchmarks/tabular_arrangement_capacity.py --full
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
