# Benchmarks

This page collects the reproducible performance and accuracy numbers for
omnibias. It is intentionally vendor-neutral: it records the *class* of
hardware used (CPU core counts, GPU memory tiers) rather than any specific
cluster, scheduler, or host.

There are two tiers of result:

- **CPU smoke tier** — small problem sizes that run on a laptop/CI in
  seconds. These are reproduced by the public scripts in
  [`benchmarks/`](https://github.com/derivon-ai/omnibias/tree/main/benchmarks)
  (committed JSON under [`docs/benchmarks/`](https://github.com/derivon-ai/omnibias/tree/main/docs/benchmarks)) and by the package
  test suites. They verify *correctness* (closed-form derivatives match
  autodiff / finite differences, cross-backend parity) **and** publish the
  CPU performance numbers the README quotes.
- **GPU headline tier** — full-fidelity training runs (FermiNet-class
  ansatze, 2D/3D PINNs, QPINN eigenstates). These require a GPU job and
  are run off-band; the numbers are transcribed here as plain text and are
  clearly labelled as such.

## Hardware classes

| Class | Description |
|---|---|
| CPU-dev | A single multi-core x86-64 CPU, no GPU (development host / CI). |
| GPU-8G | One data-center GPU with >= 8 GB memory. |
| GPU-20G | One data-center GPU with >= 20 GB memory (Laplacian-scaling sweep). |
| GPU-40G | One data-center GPU with >= 40 GB memory (FermiNet-class runs). |

## Correctness (CPU smoke tier)

These are verified in CI on every push (counts captured on a CPU-dev host):

| Check | Where | Result |
|---|---|---|
| Cross-backend parity suite | `tests/` | 509 passed (78 skipped) |
| torch <-> jax <-> keras activation parity (all orders) | `tests/test_keras_parity.py` | 48 passed / backend (rtol <= 1e-6, float64) |
| Keras unified backend (torch / JAX) | `packages/omnibias-keras/tests` | 81 passed / backend |
| Laplacian vs folx / `jax.hessian` / torch (CPU) | [`benchmarks/laplacian_scaling.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/laplacian_scaling.py) | see [Complexity](complexity.md); JSON in [`laplacian_scaling.json`](benchmarks/laplacian_scaling.json) |
| Polylaplacian `Δᵏ` vs nested (CPU) | [`benchmarks/polylaplacian_order.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/polylaplacian_order.py) | **4,660×** vs folx-nested at `k=4`; JSON in [`polylaplacian_order.json`](benchmarks/polylaplacian_order.json) |
| 1-D Poisson optimiser bake-off (CPU, 5 seeds) | [`benchmarks/optimizer_pinn.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/optimizer_pinn.py) | GN **5×** faster wall-clock than L-BFGS at matched accuracy; JSON in [`optimizer_pinn.json`](benchmarks/optimizer_pinn.json) |
| DeepONet FD accuracy floor (deterministic) | [`benchmarks/operator_fd_floor.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/operator_fd_floor.py) | 5-point `u_xxxx` floor **4.5e-6** vs closed-form **2e-17**; truncation order ≈ 2.0; JSON in [`operator_fd_floor.json`](benchmarks/operator_fd_floor.json) |
| DeepONet shared-grid residual scaling | [`benchmarks/operator_shared_grid.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/operator_shared_grid.py) | speedup → **~24×** at `F=32` with round-off agreement; JSON in [`operator_shared_grid.json`](benchmarks/operator_shared_grid.json) |
| Residual-weight / FD-step calibration | [`benchmarks/operator_residual_calibration.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/operator_residual_calibration.py) | picks Burgers `λ,n_times` and KS `h` from short-budget curves; JSON in [`operator_residual_calibration.json`](benchmarks/operator_residual_calibration.json) |
| DeepONet Burgers bake-off (CPU, 8 seeds) | [`benchmarks/operator_deeponet.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/operator_deeponet.py) | see § DeepONet; JSON in [`operator_deeponet.json`](benchmarks/operator_deeponet.json) |
| DeepONet KS bake-off (CPU, 8 seeds) | [`benchmarks/operator_ks_bakeoff.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/operator_ks_bakeoff.py) | see § KS; JSON in [`operator_ks_bakeoff.json`](benchmarks/operator_ks_bakeoff.json) |
| DeepONet vs FNO (CPU, matched steps) | [`benchmarks/operator_fno_vs_deeponet.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/operator_fno_vs_deeponet.py) | see § DeepONet; JSON in [`operator_fno_vs_deeponet.json`](benchmarks/operator_fno_vs_deeponet.json) |
| Causal marching (heat + Krishnapriyan reaction) | [`benchmarks/causal_marching.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/causal_marching.py) | equal-budget whole / causal / marching / combined on two families; smoke [`causal_marching_smoke.json`](benchmarks/causal_marching_smoke.json); full [`causal_marching.json`](benchmarks/causal_marching.json) |
| CCF DeepMind neural reproduction (phase 0) | [`benchmarks/reproduce_deepmind_ccf.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/reproduce_deepmind_ccf.py) / [`benchmarks/deepmind_campaign_tick.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/deepmind_campaign_tick.py) | Neural compactified Ω + spectral Hilbert + Martens–Grosse; dense residual gated on neural profile; stretch \(10^{-13}\) unearned (smoke ~\(10^{-1}\)); [`reproduce_deepmind_ccf_smoke.json`](benchmarks/reproduce_deepmind_ccf_smoke.json) |
| CCF Hardy line discovery + CAP | [`benchmarks/ccf_line_discovery.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/ccf_line_discovery.py) / [`benchmarks/ccf_hardy_rung_acceptance.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/ccf_hardy_rung_acceptance.py) | After stretch: exact-JVP Martens–Grosse + QR on Hardy-Ω (Adam forbidden). Absolute Rung-1/Rung-2 earned only when measured; [`ccf_line_smoke.json`](benchmarks/ccf_line_smoke.json) |
| IPM / Boussinesq scaffold | [`benchmarks/ipm_boussinesq_scaffold_smoke.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/ipm_boussinesq_scaffold_smoke.py) | Pipeline adapters + scaffold residual gates; not NS; [`ipm_scaffold_smoke.json`](benchmarks/ipm_scaffold_smoke.json) / [`boussinesq_scaffold_smoke.json`](benchmarks/boussinesq_scaffold_smoke.json) |
| SDF hard-BC geometry | [`benchmarks/geometry_sdf.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/geometry_sdf.py) | disk / annulus / CSG / nonconvex vs soft-penalty; smoke [`geometry_sdf_smoke.json`](benchmarks/geometry_sdf_smoke.json); full [`geometry_sdf.json`](benchmarks/geometry_sdf.json) |
| Parametric DeepONet zero-shot | [`benchmarks/operator_zero_shot.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/operator_zero_shot.py) | shared-IC ν sweep; conditioned vs unconditioned vs residual PINN; smoke [`operator_zero_shot_smoke.json`](benchmarks/operator_zero_shot_smoke.json); full [`operator_zero_shot.json`](benchmarks/operator_zero_shot.json) |
| Spectral bias / FBPINN | [`benchmarks/spectral_bias_fbpinn.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/spectral_bias_fbpinn.py) | equal-param arms + one-shot `lstsq`; smoke [`spectral_bias_fbpinn_smoke.json`](benchmarks/spectral_bias_fbpinn_smoke.json); full [`spectral_bias_fbpinn.json`](benchmarks/spectral_bias_fbpinn.json) |
| Fisher pack degeneracy (04-01 G2) | [`benchmarks/information_geometry.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/information_geometry.py) | two-bias logistic pack: `G_{delta,delta}` exponent `2.00 +- 0.02` over three decades, prefactor `1/720`; smoke [`information_geometry_smoke.json`](benchmarks/information_geometry_smoke.json); full [`information_geometry.json`](benchmarks/information_geometry.json) |
| Scan localization scaling (05-01 G7) | [`benchmarks/inverse_imaging.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/inverse_imaging.py) | locally-seeded logistic bias-scan: `sd(tau_hat) ~ alpha^(n - 5/2)` for `n in {3, 4}` over 1.2 decades / 5 seeds; global search earned for n=3 only; smoke [`inverse_imaging_smoke.json`](benchmarks/inverse_imaging_smoke.json); full [`inverse_imaging.json`](benchmarks/inverse_imaging.json) |
| Tabular arrangement vs LightGBM (05-02 G1/G2) | [`benchmarks/tabular_arrangement.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/tabular_arrangement.py) | Fair protocol: train on Xtr, early-stop on Xva, score Xte (no train+val refit) for both arms; H=2 soft arrangement beats tuned LightGBM by ≥10 pts on constructed oblique XOR; within 2 pts on axis AND; smoke [`tabular_arrangement_smoke.json`](benchmarks/tabular_arrangement_smoke.json); full [`tabular_arrangement.json`](benchmarks/tabular_arrangement.json) |
| Tabular arrangement public (05-02 G3) | [`benchmarks/tabular_arrangement_public.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/tabular_arrangement_public.py) | Same fair early-stop protocol; eight public binary datasets; full win/loss table (no aggregate-only headline); trees expected to win most; G4 diagnostic reported frozen; smoke [`tabular_arrangement_public_smoke.json`](benchmarks/tabular_arrangement_public_smoke.json); full [`tabular_arrangement_public.json`](benchmarks/tabular_arrangement_public.json) |
| Tabular arrangement capacity (05-02 G3b) | [`benchmarks/tabular_arrangement_capacity.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/tabular_arrangement_capacity.py) | G3 frozen; parallel capacity/optimizer arms vs the same LightGBM; primary arm `boost_h2` predeclared; G3b unearned (`boost_h2` not-worse 4/8, need >=6/8); smoke [`tabular_arrangement_capacity_smoke.json`](benchmarks/tabular_arrangement_capacity_smoke.json); full [`tabular_arrangement_capacity.json`](benchmarks/tabular_arrangement_capacity.json) |
| Multipack Birkhoff (01-01 G1/G2/G3/G5) | [`benchmarks/multipack_birkhoff.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/multipack_birkhoff.py) | closed-form multi-pack vs mpmath with recorded float64 order ceiling; FD collapse stability; torch/jax parity; poisedness honesty; smoke [`multipack_birkhoff_smoke.json`](benchmarks/multipack_birkhoff_smoke.json); G4 deferred |
| Irregular Birkhoff stencils (01-04 G1–G4) | [`benchmarks/irregular_stencils.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/irregular_stencils.py) | exact-`Q` weights reproduce `signs_exact`; empirical rate vs reported accuracy; certificate covers grid+sample; poisedness oracle; smoke [`irregular_stencils_smoke.json`](benchmarks/irregular_stencils_smoke.json) |
| Bias scan (01-02 G1/G2/G3; G4 smoke-earned) | [`benchmarks/bias_scan.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/bias_scan.py) | interior-shift equivariance ≤ 4 ulp; 5-seed localization MAE ≤ 0.1 spacing with skill vs midpoint; torch/jax within 4 ulp; two-interface soft-argmax bias recorded (not a win); G4 no-grid vs voxelized `cmbConv1d` earned on smoke, **not** in CI `all_passed` (wall time); smoke [`bias_scan_smoke.json`](benchmarks/bias_scan_smoke.json) |
| Mollifier calculus (01-05 G1–G3; G4 deferred) | [`benchmarks/mollifier_calculus.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/mollifier_calculus.py) | closed-form moments; order-`m` polynomial exactness + `O(eps^m)`; tail bound 0 violations; analytic bases have certified exponential tails, not compact support; G4 deferred to VPINN; smoke [`mollifier_calculus_smoke.json`](benchmarks/mollifier_calculus_smoke.json) |
| Spectral design (01-07 G1/G2/G4; G3 not CI-gated) | [`benchmarks/spectral_design.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/spectral_design.py) | peak vs numerical argmax ≤ 1e-6; transfer magnitude within 2%; hole-detection; pack order is a band selector; 01-06 frames stay concept; G3 not in CI `all_passed`; smoke [`spectral_design_smoke.json`](benchmarks/spectral_design_smoke.json) |
| Scan-Net (02-01 G1/G2/G5; G3/G4 recorded) | [`benchmarks/scannet.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/scannet.py) | interior-shift ≤ 4 ulp in the stack; scattered two-interface vs voxelized `cmbConv1d` with skill vs midpoint; torch/jax bit-identical; equivariance is on-lattice, not `R^D`; G3 wall/point vs `N` and G4 k-NN (k-NN may win) recorded, **not** in CI `all_passed`; smoke [`scannet_smoke.json`](benchmarks/scannet_smoke.json) |
| Jet-KAN (02-03 G1/G3/G5; G2 smoke-earned) | [`benchmarks/jetkan.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/jetkan.py) | model-jet vs inlined cubic B-spline; exactness is of the model jet, not the target; KA theorem does not justify; G2 jet-vs-autodiff timing **not** in CI `all_passed`; smoke [`jetkan_smoke.json`](benchmarks/jetkan_smoke.json) |
| Weak-form VPINN (02-04) | [`benchmarks/weak_form_vpinn.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/weak_form_vpinn.py) | exact vs Gauss on polynomial boxes; boundary bound on by default; discontinuous-coeff manufactured 1-D; smoke [`weak_form_vpinn_smoke.json`](benchmarks/weak_form_vpinn_smoke.json) |
| Multi-interface transmission PINN (02-05) | [`benchmarks/multi_interface_pinn.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/multi_interface_pinn.py) | sharp-limit piecewise poly; residual ~`1/alpha`; `alpha -> inf` is interface sharpening, neither collapse; parallel interfaces only; smoke [`multi_interface_pinn_smoke.json`](benchmarks/multi_interface_pinn_smoke.json) |

Four-gap status matrix (capability / empirical / structural / certified):
[`benchmarks/pinn_four_gap_matrix.md`](benchmarks/pinn_four_gap_matrix.md).

## Performance (GPU headline tier)

!!! note
    The full-fidelity GPU numbers are produced off-band on GPU-class
    hardware and transcribed here. Where a number is not yet transcribed
    it is marked *pending*. The public, regenerable CPU equivalents live in
    [`benchmarks/`](https://github.com/derivon-ai/omnibias/tree/main/benchmarks)
    and are the numbers the README quotes under "reproduce this".

| Workload | Hardware class | Metric | Result |
|---|---|---|---|
| Local-kinetic-energy Laplacian, closed-form vs `jax.hessian` (`D=240`) | GPU-20G | speedup / memory | **67.7×** time, **63×** memory |
| Local-kinetic-energy Laplacian, closed-form vs `torch.func.hessian` (`D=240`) | GPU-20G | speedup / memory | **198.7×** time, **108×** memory |
| Local-kinetic-energy Laplacian, closed-form vs folx (`D` 3→240) | GPU-20G | speedup | ~1.0–1.25× (tie; both flat in `D`) |
| Polylaplacian `Δ³`, closed-form vs folx-nested (`D=60`) | GPU-20G | speedup | **518×** (folx-nested OOMs at `Δ⁴`) |

All four methods agree to `<= 1e-15` (float64); see the full sweep and the
time/memory **complexity derivation** in [Complexity](complexity.md). Conditions:
`H=256`, `B=4096`, float64, single data-center GPU.

| Workload | Hardware class | Metric | Result |
|---|---|---|---|
| 2D Navier-Stokes PINN | GPU-8G | residual at convergence | *pending* |
| QPINN harmonic oscillator (TISE) | GPU-8G | energy error vs analytic | *pending* |
| QPINN dark soliton (Gross-Pitaevskii) | GPU-8G | norm drift | *pending* |

## Curvature optimizers (PINN / operator / scaling / LLM)

`omnibias.torch.optim` / `omnibias.jax.optim` ship exact-curvature optimisers as
drop-in tools alongside Adam (nothing is taken away). The DeepMind-style
singularity earn path uses **damped Gauss–Newton with Martens–Grosse closed-form
LR/momentum via exact JVPs** (`martens_grosse_combine`) and a **non-squaring QR
(or CGLS) LM solve** — not finite-difference MG probes and not dense
\(J^\top J\) as the default. Three vendor-neutral studies, all
iso-wall-clock against a tuned Adam/AdamW baseline that is never handicapped:

**1. PINN accuracy (6 PDEs, CPU-dev, float64, 5 seeds).** Best relative-L2 vs the analytic solution
(median of 5 seeds); fair timing requires pinning BLAS/torch threads to the reserved core count. The
exact Gauss–Newton method (`cubic_gauss_newton`) wins every PDE, and the matrix-free `trust_newton` /
`stochastic_newton` are consistently top-tier and always beat Adam:

| PDE | `cubic_gauss_newton` | `trust_newton` | tuned Adam |
|---|---|---|---|
| 1-D Poisson | 1.5e-8 | 3.6e-8 | 1.5e-5 |
| 1-D Helmholtz (stiff) | 6.7e-5 | 7.2e-4 | 8.3e-3 |
| 2-D heat | 2.4e-5 | 1.6e-4 | 9.3e-4 |
| Burgers | 1.5e-4 | 3.1e-4 | 1.1e-3 |
| Allen–Cahn | 3.6e-4 | 7.4e-4 | 3.2e-3 |
| 2-D Kovasznay NS | 6.5e-4 | 9.9e-4 | 1.7e-3 |

Honest caveat: the *diagonal* "Adam substitutes" only match Adam and fail the stiff Helmholtz case — a
diagonal cannot capture the operator's off-diagonal coupling. The wins are the exact full / Gauss–Newton
methods, at small-to-medium PINN scale.

**2. Matrix-free scaling (GPU, 1e3–1e6 parameters, float64).** A Newton step never forms the Hessian;
it applies `v ↦ Hv` by double-backward autodiff, so per-step memory is `O(P)`. Sweeping a `JetMLP` PINN
and a supervised teacher–student MLP from ~1e3 to ~1e6 parameters, peak GPU memory grows *linearly* —
`trust_newton` peaks at **~210 MB at 1e6 parameters**, where a *dense* `float64` Hessian would need
`O(P²) ≈ 8 TB` (a ~38,000× gap; it simply would not fit). The mean Steihaug-CG count per step stays
roughly flat (~8–10 for `trust_newton`, ~15 for `stochastic_newton`) across three orders of magnitude
in `P` — that flatness is *why* the step scales. On the smooth Poisson operator `trust_newton` drives
rel-L2 to `~1e-5`, roughly three orders of magnitude below tuned Adam (~1e-2), while using less than
half the peak memory of `torch.optim.LBFGS` (210 MB vs 490 MB); on the supervised regression the exact
`trust_newton` / `stochastic_newton` again reach the best error (~2e-4) at the lowest memory. Honest
boundary: on the stiff high-wavenumber Helmholtz case L-BFGS's accumulated history wins within a fixed
step budget. This is the evidence that the exact second-order step is usable at real network sizes.

**3a. Operator learning — 1-D Fourier Neural Operator (GPU).** A compact FNO learning the viscous-Burgers
solution operator `u(·,0) ↦ u(·,T)` extends the iso-wall-clock bake-off to operator learning; the exact HVP
flows through torch's complex-FFT autograd. The interesting result is that the verdict depends on *what you
ask the operator to fit* — the same optimisers flip from loser to winner when the objective is reshaped.
(K-FAC is N/A throughout: the spectral-conv weights are not `nn.Linear`.)

*Plain data regression* (the original honest loss) — supervise `u(·,0) ↦ u(·,T)` only. **Tuned Adam wins by
~5×**; the full-batch second-order methods plateau ~5× higher at equal wall-clock. The operator-regression
loss is ill-conditioned and flat (spectral bias), so Adam's many cheap steps beat a few expensive exact
steps — a genuine "second-order does not always win" result. Here the methods do *not* overfit
(train ≈ test); they simply reach a worse minimum.

| FNO / Burgers, data-only (3 seeds) | best held-out rel-L2 | steps→tol | wall |
|---|---|---|---|
| tuned Adam | **6.0e-3** | 1080 | 22 s |
| `cubic_gauss_newton` | 3.0e-2 | — | 92 s |
| `stochastic_newton` | 3.3e-2 | — | 90 s |
| `trust_newton` | 3.5e-2 | — | 90 s |
| `torch` L-BFGS | 1.3e-1 | — | 90 s |
| `diag_hutchinson` | ≈0.9 (failed) | — | 19 s |

*Physics-informed reshaping (PINO)* — add the viscous-Burgers residual `u_t + u·u_x − ν·u_xx` to the loss
(`u_x`, `u_xx` spectral via the data generator's own `irfft(ik·rfft(u))` convention; `u_t` by an endpoint
finite difference). The residual makes the geometry curvature-friendly and **the verdict flips**: at *true
iso-wall-clock* (300 s per method, with Adam's step cap lifted so it is never handicapped), the exact-curvature
methods reach a **~1.5× lower best held-out rel-L2** than tuned Adam, in far fewer steps. `trust_newton` is
robust across the PDE-weight sweep (`w=10, 30`); `cubic_gauss_newton` matches it at `w=10` but degrades at
very high weight.

| FNO / Burgers, FD-PINO `w=10` (iso-wall 300 s, median of 3–4 seeds) | best held-out rel-L2 | steps | wall |
|---|---|---|---|
| `trust_newton` | **5.0e-2** | ~8.9k | 300 s |
| `cubic_gauss_newton` | 5.2e-2 | ~1k | 300 s |
| tuned Adam | 8.0e-2 | ~60k | 300 s |

(At `w=30` the picture is the same for `trust_newton` — 5.3e-2 vs Adam 8.9e-2, a 1.7× win — while `cubic_gauss_newton` degrades to 1.5e-1 under the heavier residual.)

Honest caveats — the win is real but modest and requires early stopping:

- **Both** Adam and the curvature methods *overfit the residual* if run past their early minimum (best rel-L2
  is reached in ~30–80 steps for Gauss–Newton, then test rel-L2 climbs back to ~0.11–0.18). The number above is
  the best (early-stopped) checkpoint, not the asymptotic one.
- The ~1.5× is measured at true iso-wall. An earlier iso-*budget* run reported ~1.7–1.9× but had under-walled
  Adam (Adam hit a 3k-step cap at ~33 s while curvature ran ~130 s); lifting Adam's cap for a fair 300 s shrinks
  the gap to ~1.5×. We report the fair number.
- The win is on the *median*: the curvature methods have a bad-seed tail (an occasional seed fails to rel-L2
  ~0.2–1.0), so their variance is higher than Adam's. `trust_newton` is the most robust across PDE weights.
- The *time-conditioned* PINO variant (exact `u_t` via an added time channel + regenerated multi-snapshot data)
  is implemented, but exact `u_t` + curvature costs ~50–100× Adam's per-step wall here, so it is compute-limited
  at this budget; the finite-difference PINO is the headline and the time variant is left as an honest
  "compute-bound, not yet conclusive" note.
- `JetSubspaceTensor` is wired into the FNO/PINO bench but is *not* a headline here: its order-3 tensor needs
  `O(k³)` directional third derivatives through the spectral-conv complex-FFT autograd, which is even costlier
  than the time-conditioned variant above, so at this budget it is compute-bound with no iso-wall advantage over
  the matrix-free `trust_newton` (the FNO/PINO pick). The subspace-Newton fallback (`order=2`) is the sensible
  configuration if it is used on operator learning at all.

Net: reshaping the objective flips a 5× *loss* into a ~1.5× iso-wall *win* — evidence that the honest data-only
FNO result was about problem geometry, not the optimiser.

**3b. GPT-2-small curvature demo (GPU, WikiText-103).** A GPT-2 built entirely from `nn.Linear` compares
tuned AdamW against exact-HVP curvature methods at equal wall-clock, sharing model / batch / precision /
budget; AdamW is tuned and never handicapped. The headline is 124M parameters; a fast **30M** preset is used
to iterate across methods and regimes.

*124M headline (final, ≤200-min iso-wall, single seed).* All three methods share model / batch / precision /
budget; AdamW is tuned and never handicapped, and the K-FAC hybrid (K-FAC on the per-block attention / MLP
linears, AdamW on embeddings / LayerNorm / tied head) runs the **stabilized** package optimiser.

| GPT-2 124M / WikiText-103 (≤200 min, 1 seed) | best val ppl | best @ step | ppl @ 35k (iso-step) | steps in budget |
|---|---|---|---|---|
| tuned AdamW | **33.3** | 76k | **39.7** | 91k |
| K-FAC hybrid (stabilized) | 38.8 | 37k | 41.3 | 39k |
| Shampoo (internal) | 63.4 | 34.5k | 65.3 | 35k |

Three honest findings at 124M:

- **Iso-wall, AdamW wins** (best ppl 33.3). No curvature method beats it on wall-clock at this scale.
- **The 30M per-step Shampoo edge did _not_ scale.** At 124M Shampoo trails badly (best 63.4): its
  per-2D-weight eigendecompositions are far costlier relative to the larger matmuls, so it completes only ~35k
  steps and is behind at every matched step. Treat the 30M per-step win below as a small-model curiosity.
- **Stabilized K-FAC scales — stable and per-step competitive.** It reaches 38.8 with **no divergence and no
  loss spikes** (vs the first-gen 101.9 below), and per step it *catches* AdamW: by ~39k steps K-FAC is 39.1 vs
  AdamW 39.2 — a per-step tie, not a win, but a decisive confirmation that the hardening holds at 124M.

*Pre-stabilization reference (124M).* The earlier run that motivated the hardening: the first-generation K-FAC
hybrid was slow per step and prone to divergence, and a Hutchinson-diagonal control was not competitive.

| GPT-2 124M / WikiText-103 (≤200 min, pre-stabilization) | best val perplexity | steps in budget | tok/s |
|---|---|---|---|
| tuned AdamW | 32.6 | 100k (step cap) | 74.7k |
| K-FAC hybrid (first-gen) | 101.9 | 30.5k | 17.9k |
| Hutchinson diagonal | 504.5 | 100k (step cap) | 82.2k |

First-gen K-FAC ran ~4× slower (17.9k vs 74.7k tok/s), completed only ~30k steps, and showed the transient loss
spikes characteristic of K-FAC when a factor becomes ill-conditioned — exactly the failure mode the
tracked-package hardening (adaptive Levenberg damping with accept/reject + a trust-region step-norm cap +
optional Cholesky solver, all new backward-compatible knobs in `omnibias.torch.optim.KFAC`) removes.

*Stabilized + extended methods (30M preset, ~70-min budget, single seed).* The K-FAC hybrid now runs
**stabilized** — adaptive Levenberg damping (accept/reject) + a trust-region step-norm cap + an optional
Cholesky solver, all new backward-compatible knobs in `omnibias.torch.optim.KFAC` — and no longer spikes.
We also prototype **Shampoo** and **SOAP** (Kronecker-factored, Adam-grafted; internal-experimental) and an
exact-diagonal **Sophia**.

| GPT-2 30M / WikiText-103 (~70 min, 1 seed) | best val ppl | ppl @ 25k steps | steps in budget |
|---|---|---|---|
| tuned AdamW | **43.0** | 47.0 | 67.8k |
| Shampoo (internal) | 44.3 | **44.5** | 28.2k |
| K-FAC hybrid (stabilized) | 46.4 | 56.5 | 56.6k |
| exact-diagonal Sophia | 661 (failed) | — | 60k |
| SOAP (internal) | NaN (diverged) | — | — |

Two honest findings:

- **Per-step (iso-step) at 30M, Shampoo wins.** For steps ≥ 20k it reaches a lower perplexity than AdamW at the
  same step (44.5 vs 47.0 at 25k) and hits its best in ~28k steps vs AdamW's ~56k. **This edge is scale-specific:
  it does _not_ survive to 124M** (headline above), where Shampoo trails throughout — so read it as a small-model
  curiosity, not a general per-step win for a curvature-structured method.
- **Iso-wall, AdamW wins.** Shampoo's per-2D-weight eigendecompositions cost ~2.4× per step, so in the fixed
  budget AdamW takes ~68k steps to Shampoo's ~28k and reaches a marginally better absolute best (43.0 vs 44.3).
  Stabilized K-FAC no longer diverges but warms up slowly (factor accumulation) and lands at 46.4. SOAP is
  harder to stabilize than Shampoo (rotating eigenbasis; still diverged here) and the exact-diagonal Sophia did
  not train competitively at the learning rates tried — both reported as honest negatives.

*Regime study (30M): where does curvature pay?* Measured at the best (early-stopped) checkpoint, **AdamW still
wins both** regimes — but the failure modes are informative:

| 30M regime (best val ppl) | tuned AdamW | K-FAC hybrid (stabilized) |
|---|---|---|
| small-batch (batch 2) | **137** | 458 |
| low-data (3M-token cap) | **154** | 288 |

- **Small-batch hurts K-FAC**, as expected: batch-2 activations give noisy Kronecker factors, so the
  preconditioner degrades. Curvature-from-statistics wants a reasonable batch.
- **Low-data is nuanced:** AdamW reaches a better *early* minimum (154 vs 288) but then overfits the 3M-token cap
  catastrophically (final val ppl **3394**), whereas stabilized K-FAC resists it (final **493**, ~6.9× lower).
  So curvature buys robustness to late-stage overfitting when data is scarce — a real effect, not a
  best-checkpoint win.

Takeaway (as anticipated): exact curvature is a decisive tool for smooth, full-batch scientific-ML objectives
(PINNs §1–2; reshaped operator learning §3a). At LM scale the 124M headline confirms the honest picture:
**tuned AdamW wins wall-clock (33.3)**, the 30M Shampoo per-step edge does not replicate, and **stabilized K-FAC
scales cleanly — 38.8 with no divergence or loss spikes, tying AdamW per step**. That is a stability-and-scaling
result for the hardened package optimiser, not a SOTA claim.

**3c. Off-PINN control — plain MLP / CNN, a fair GPU bake-off (A100).** PINNs and operator learning share two
properties: a *smooth* objective and one that *is* the learning target (minimising the residual solves the PDE).
The `cnn_mlp_bench.py` reproduction (in the separate `omnibias_experiments` project) separates the axes this conflates — **architecture** (a plain `MLP`
vs a `CNN`) × **regime** (a smooth full-batch regression vs a mini-batched classification) — on tiny, seeded,
realizable teacher–student tasks. Every matrix-free method is architecture-agnostic (it needs only HVPs, so it
applies to the conv net); K-FAC is `nn.Linear`-only. Run on one A100 (float64, CUDA-synchronised **iso-wall 20 s**,
median of 5 seeds) so the conv HVP is timed fairly — the shared CPU node could not (NNPACK-less conv
double-backward + contention gave seconds/step and made even Adam's step count swing 4×).

*Smooth full-batch regression — optimisation speed.* The honest optimiser metric is how fast it drives the
training objective down. On the MLP we log steps / wall to reach MSE ≤ 1e-4; the tiny CNN reaches no method to
1e-4 within 20 s, so we show the objective and step count reached instead.

| regression (median / 5) | best train MSE | held-out rel-L2 | steps → 1e-4 | wall → 1e-4 | total steps |
|---|---|---|---|---|---|
| **MLP** — tuned Adam | 1.7e-6 | 0.357 | 2490 | 7.8 s | 6600 |
| MLP — L-BFGS | 5.4e-5 | 0.358 | 410 | 17.2 s | 470 |
| MLP — trust-region Newton-CG | 2.9e-6 | 0.371 | **40** | 6.9 s | 79 |
| MLP — cubic Gauss-Newton | 3.9e-6 | 0.390 | 50 | 9.4 s | 159 |
| MLP — `JetSubspaceTensor` (order 3) | 1.5e-4 | 0.393 | — | — | 577 |
| MLP — K-FAC | 8.7e-3 | 0.356 | — | — | 720 |
| **CNN** — tuned Adam | 2.1e-3 | 0.280 | — | — | 1690 |
| CNN — L-BFGS | 2.6e-3 | **0.249** | — | — | 370 |
| CNN — trust-region Newton-CG | **1.6e-3** | 0.303 | — | — | 51 |
| CNN — cubic Gauss-Newton | 2.4e-3 | 0.306 | — | — | 37 |
| CNN — `JetSubspaceTensor` (order 2) | 2.1e-3 | 0.283 | — | — | 292 |

(SGD and the Hutchinson diagonal never reach 1e-4 and are omitted for space; both trail on the objective.)

- **Iso-step, curvature wins by 35–60× — on both architectures.** Trust-region Newton-CG reaches the MLP's 1e-4
  in **40 steps vs Adam's 2490**; on the CNN it reaches a *lower* objective than Adam's full 20 s (1.6e-3 vs
  2.1e-3) in **51 steps vs ~1700**. The exact-curvature per-iteration advantage is real, large, and
  architecture-agnostic — the same effect that dominates PINNs, now with no physics.
- **`JetSubspaceTensor` — the exact 3rd-order option, and an honest negative on smooth nets.** It builds a small
  Krylov subspace and minimises the *exact* degree-3 Taylor model there (order 2 = subspace Newton). It beats
  first-order Adam iso-step (a ≥10× lower loss at equal step count is a gated regression test), but on these
  *near-quadratic* smooth objectives the cubic term buys nothing while its `O(k³)` directional third derivatives
  make it the **most expensive** curvature step (577 MLP steps in 20 s ≈ 35 ms/step). It is therefore
  iso-wall-dominated by the matrix-free trust-region Newton-CG / cubic Gauss-Newton, which reach a far lower
  objective in far fewer steps. This confirms the design's own caveat: use `order=2` when wall-clock is the
  metric, and reserve the order-3 tensor for objectives whose cubic curvature actually matters. It is a correct,
  fully-gated addition to the optimiser suite — not a new default.
- **Iso-wall on these tiny nets is ≈ a wash.** Each Newton step costs ~30 HVPs, so the 60× step advantage nets to
  near-parity in wall: trust-region Newton-CG hits the MLP target in 6.9 s vs Adam's 7.8 s (a slim edge), while
  L-BFGS / cubic-GN are step-thrifty but wall-slower. The step advantage becomes a *wall* win only when steps are
  the bottleneck — an ill-conditioned objective — which is exactly the PINN regime (§1–2).
- **Generalisation is a wash, and full-Newton can overfit.** Held-out is essentially tied (MLP ~0.36; CNN
  ~0.25–0.31); the full-Newton methods fit the objective hardest yet land *slightly worse* than Adam on held-out,
  and the best CNN held-out belongs to L-BFGS (0.249). A lower objective is not a better model off the PINN
  objective — unlike a PINN, where by construction it is.
- **K-FAC** (MLP only) is fast early but its approximate natural gradient plateaus (8.7e-3, never 1e-4); held-out
  still ≈ Adam.

*Mini-batch classification — held-out error rate (median / 5).*

| mini-batch classification | MLP | CNN |
|---|---|---|
| tuned Adam | **0.213** | 0.244 |
| momentum SGD | 0.216 | 0.242 |
| Hutchinson diagonal | 0.221 | **0.240** |
| K-FAC (`nn.Linear`) | 0.301 | — |
| subsampled Newton-CG | 0.231 | 0.278 |

- **Tuned Adam / SGD win on both architectures** (with the cheap Hutchinson diagonal tied on the CNN); the full
  subsampled-Newton step is noisier — its curvature comes from a single minibatch — and trails. The
  stochastic-regime boundary is architecture-independent.

Takeaway (fair GPU numbers): the omnibias curvature advantage is **objective-driven, not architecture-driven** —
a large per-iteration win that converts to a *wall-clock* win only on ill-conditioned, full-batch,
objective-*is*-target problems (PINNs / operator residuals). On ordinary MLP/CNN supervised training the
matrix-free methods are correct and 35–60× more step-efficient, but iso-wall-neutral on small nets and
generalisation-neutral, so tuned Adam/SGD remain the right default. In one line: **the best default for PINN-type
objectives; a strong optional optimiser (not a new default) for general MLP/CNN training.**

**3d. Proof-carrying training — a trust deliverable, not a speed one.** The benches above ask *how fast* an
optimiser drives the loss down. `omnibias.verify.certify_trained_min` asks a different, orthogonal question with a
**proof**: is a trained `θ*` a genuine, locally-unique, *strict* local minimum of the training loss, or just a
point where the optimiser stopped? Over a parameter ball `B(θ*, radius)` it proves a locally-unique stationary
point (Krawczyk on `∇_θ L`) **and** a positive-definite Hessian (interval `LDLᵀ` inertia with a positive-definite
shift giving a sharp `eig_min.lo > 0`), then seals a tamper-evident v1 certificate whose scalar `eig_min > 0`
obligation the Lean kernel can re-check. On a realizable single-`tanh`-unit fit (`P = 4`, 12 data points) it
certifies a strict minimum with `eig_min ∈ [0.026, 0.16]` at `radius = 1e-3` in well under a second; two negative
controls (a non-stationary point and a too-wide ball) are correctly *refused*, and the interval gradient / Hessian
are checked sound against an independent finite-difference reference on a dense grid plus random in-box samples.
Certification reaches past a single unit through **regularisation**, not better arithmetic: an over-parametrised
2-unit net (`P = 7`) has a near-flat *bare*-loss minimum (smallest true Hessian eigenvalue `~1e-7`, so it is not
strict and is honestly refused), but the minimiser of the L2-regularised objective `J = L + 0.2·‖θ‖²` — whose
Hessian is lifted by `2·l2` (folded in exactly) — is certified strict with `eig_min.lo ≈ +0.11` at `radius = 1e-3`.
(An affine / zonotope enclosure engine was implemented and measured for this and then *rejected*: the shallow,
cancellation-free parameter-space Hessian sweep has no wrapping for it to fight, so it only widened the bound
~1.4×.) This is a rigorous **local** proof for a small net with fixed data — the documented interval-method scope —
not a global-optimality or global-regularity-grade claim. See the [omnibias-verify page](api/verify.md#proof-carrying-training-a-strict-local-minimum-certificate).

**3e. Proof-carrying optimization — fusing the subspace tensor with the rigorous core.**
`certify_trained_min` (3d) certifies where an optimiser *stopped*; `omnibias.verify.certify_subspace_step` certifies
each step it *takes* — the one primitive that turns the two training bets into more than the sum of their parts.
It takes the *exact* degree-3 Taylor model of the loss on a small Krylov trust region (the same object
`JetSubspaceTensor` minimises — 3b) and, in the rigorous register, encloses it as an order-3 `TaylorModelMV` whose
scalar remainder `R` bounds the model-vs-truth error over the *whole* ball (the activation towers composed in closed
form via the verified `sigma_tower_interval` + a Makino–Berz remainder). The true per-step decrease is then enclosed
by `[pred − w, pred + w]` (`pred = m(0) − m(a*)`, `w = R.width`), so the step is **certified to strictly decrease the
true loss** whenever `pred − w > 0` — and since `R = O(r⁴)` while `pred = O(r)`, a small enough radius always
certifies. The margin's `> 0` sign obligation is Lean-checkable. On the realizable `1-1-1` fit it certifies descent
for `r ≤ 0.2` (a 6-step trajectory drives the loss `0.999 → 0.714`, every step proof-carrying and monotone), the
enclosed decrease lower bound is confirmed against the true float drop, and a cross-package test feeds a torch
`taylor_subspace_model` basis + `solve_subspace_trust_region` step straight into the pure-Python certifier — the
differentiable and rigorous registers agreeing on `(c, H)`, then the rigorous one certifying the differentiable
optimiser's own move. This is *not* a wall-clock deliverable (it competes with 3b/3d on trust, not speed): it is the
novel **proof-carrying optimization trajectory** that neither bet had alone. See the
[omnibias-verify page](api/verify.md#proof-carrying-optimization-a-certified-subspace-trust-region-step).

**4. The solver driver (`omnibias.pinn.solver`, GPU-class node, float64, 5 seeds).** Study 1 benchmarks
`omnibias.torch.optim` on hand-written PINN losses; this one benchmarks the same optimisers *through*
`solve_optimize`, so it measures the wiring rather than the math. Identical setup for every method: one
`build_field` ansatz (`hidden=32`, `tanh`), a 200-iteration Adam warmup, then 50 iterations of the method
under test on a 16x16 interior / 16-point-per-face collocation grid with `condition_weight=20`. The metric is
relative L2 against the analytic solution on a held-out dense grid — never the training residual norm.

*Equal-epoch* (50 iterations each, median relative L2 over seeds 0-4):

| method | 2-D Poisson | 1-D + time heat |
|---|---|---|
| `gauss_newton` (`solver="qr"`, Nielsen damping) | **1.9e-4** | **6.6e-5** |
| `cubic_gauss_newton` | 1.4e-3 | 1.6e-4 |
| `natural_gradient` (Gauss-Newton Fisher) | 4.7e-3 | 5.8e-3 |
| `torch` L-BFGS (strong Wolfe) | 6.4e-2 | 7.3e-3 |
| tuned Adam | 1.1e-1 | 5.1e-2 |

*Equal-wall-clock*, the arm that matters: an exact Gauss-Newton iteration costs roughly 6-8x an L-BFGS
iteration here, so the baselines are re-run with their iteration count scaled up until their post-warmup
wall-clock meets or exceeds `gauss_newton`'s (0.78 s on Poisson, 0.94 s on heat). They still lose:

| method | 2-D Poisson | 1-D + time heat |
|---|---|---|
| `gauss_newton`, 50 iters | **1.9e-4** (0.78 s) | **6.6e-5** (0.94 s) |
| L-BFGS, 333 / 404 iters | 1.6e-3 (0.96 s) | 6.6e-4 (1.23 s) |
| Adam, 453 / 468 iters | 4.7e-2 (0.86 s) | 1.2e-2 (1.02 s) |

So second order wins both arms on these two problems — 8-10x at equal wall-clock, two orders of magnitude at
equal epochs — which is why `solve_optimize` accepts the curvature optimisers. Honest boundaries: the default
stays L-BFGS (cheapest per step, no `optimizer_kwargs` tuning, and study 3a shows curvature can lose outright
on an ill-conditioned operator-regression objective); `gauss_newton` needs `solver="qr"` or `"cgls"` to avoid
squaring a stiff operator's conditioning; both problems are linear and small; and the whole path is
torch-only, because the JAX solver has no `solve_optimize`. `KFAC` is not offered at all — it builds its
Kronecker factors from `nn.Linear` hooks that the closed-form jet forward never triggers.

## Adaptive collocation and inverse problems (omnibias-pinn solver)

### Residual-adaptive collocation (RAR)

Does moving collocation points toward the residual buy accuracy at an **equal final point budget**? The RAR
arm starts at 400 random interior points and grows to 1200 over 8 rounds; the uniform arm draws 1200 up
front. Same field init, same warmup, same optimiser (L-BFGS), same iteration count — the only difference is
*where* the points are. The metric is the held-out dense-grid (200²) residual **max-norm**, because the claim
RAR makes is about the worst point, not the average one. GPU-class node, float64, 8 seeds, `hidden=64`,
200 iterations after a 1500-iteration Adam warmup.

Ratios are uniform / RAR, so above 1 means RAR is better:

| problem | `greedy` | `proportional` (default) |
|---|---|---|
| sharp front (bistable, `D=0.002`, `k=20`) | **4.07x** (8/8 seeds) | 3.42x (8/8) |
| Burgers, `nu=0.003` | 1.13x (5/8) | **1.23x** (8/8) |
| Burgers, `nu=0.01` | 0.96x (3/8) | **1.10x** (6/8) |

A second sweep at a much smaller budget (16 seeds, `hidden=24`, 100 → 220 points, 40 iterations) reverses the
strategy ordering — greedy leads everywhere there: 3.05x / 1.18x / 1.17x against proportional's 2.10x / 1.07x
/ 1.08x. That reversal is the reason the default is `proportional`: across all six problem-by-budget cells its
median is never below 1.07x, while greedy posts an outright **median regression** (0.96x) on the smoothest
problem at the larger budget. Greedy is the sharper instrument when the budget is tight and the front is
narrow; proportional is the one that never loses.

RAR is also *cheaper* here, not a trade: 4.3 s versus 6.4 s on Burgers and 6.3 s versus 8.8 s on the front,
because the point set is small for the early iterations and only reaches the full budget at the end.

Honest boundaries: the win is large and per-seed only on the sharp front; on smooth-ish Burgers it is a
median effect with individual seeds losing, and the tests gate it that way rather than asserting a per-seed
win that is not there. Moving points onto a shock costs resolution elsewhere. RAR is wired into
`solve_optimize` only (`solve_least_squares` caches a `CollocationPlan`), and never runs during the Adam
warmup, where the residual reflects initialisation rather than solution structure.

### Inverse problems: coefficient recovery

Can `solve_inverse` recover a known coefficient from a 3x-wrong initial guess? Three cases chosen for
structure rather than flattery: `heat` / diffusivity (linear, analytic reference), `wave` / speed (enters as
`speed²`, so only the magnitude is identifiable), and `burgers` / viscosity (nonlinear, reference from a
forward solve pinned at the truth). 48 scattered observations, `hidden=16`, 40 iterations after a
150-iteration Adam warmup, 6 seeds, GPU-class node, float64. Relative error of the recovered coefficient,
median over seeds:

| case (truth) | `cubic_gauss_newton` (default) | `torch` L-BFGS |
|---|---|---|
| `heat`, `D = 0.35` | **0.02%** (1.6 s) | 17.2% (0.4 s) |
| `wave`, `c = 1.4` | **0.07%** (1.8 s) | 154.7% (0.4 s) |
| `burgers`, `nu = 0.08` | **2.6%** (1.5 s) | 72.2% (0.3 s) |
| two coefficients at once (`0.4`, `0.15`) | **0.16%** (4.5 s) | 19.5% (0.7 s) |

This is the largest optimiser gap anywhere in these benchmarks, and it is structural rather than a tuning
artefact: L-BFGS barely moves the coefficient at all (`wave` ends at 3.57 against a truth of 1.4 — further
from the answer than the initial guess). A lone scalar coefficient and a few hundred network weights have
curvature on completely different scales, and a method with one shared step size cannot serve both, whereas
the exact Gauss-Newton metric rescales each direction by its own curvature. That is why `solve_inverse`
defaults to `cubic_gauss_newton` even though `solve_optimize` still defaults to L-BFGS; at 4-6x the per-step
cost it remains the right default here, since the L-BFGS answer is not usable at any budget tested.

Noise sweep on `heat` (Gaussian noise as a fraction of the solution amplitude, median relative error):

| noise | 0% | 1% | 5% | 15% |
|---|---|---|---|---|
| recovered `D` error | 0.02% | 0.75% | 3.8% | 11.5% |

The degradation is roughly linear in the noise level and stays well inside it — 15% noise costs 11.5% error —
which is the expected behaviour for a well-posed least-squares recovery, and the growth itself is gated in
the tests: a fit that ignored the data would show a flat curve instead.

Honest boundaries: torch-only. Identifiability is the caller's responsibility and **fails silently** — a
coefficient the data cannot see does not raise, it just stops moving. `wave` recovers `|c|` only. All three
problems are small and their coefficients enter linearly in the residual; a coefficient inside a
nonlinearity, or one poorly excited by the observation set, is a harder problem than anything measured here.

### Hard vs soft boundary / initial conditions

A boundary condition is normally one more penalty term, weighted against the interior residual. Absorbing it
into the ansatz instead makes it exact -- that part is algebra -- but says nothing about whether the rest of the
solve improves, so both halves are measured. Poisson (elliptic), heat (parabolic), wave (hyperbolic), a 2-D
square whose four faces are absorbed at once, a gauge-free periodic seam, and a **gauge-pinned** periodic heat,
each with an analytic solution, 5 seeds, `solve_least_squares`, `hidden=96`, 48 interior / 16 per face. The MLP
hard/soft arms share architecture, parameter count, seed and collocation budget; only `hard_conditions` differs.
The gauge-pinned heat row also runs a third **spectral** arm (`basis="spectral"`, `K=8`, `hard_conditions="auto"`)
where spatial periodicity is free in the Fourier base. Regenerate with `benchmarks/hard_conditions_solver.py`.

| problem | absorbed | boundary violation (hard, **max** over cells) | boundary violation (soft, median) | interior rel-L2 hard | interior rel-L2 soft | hard wins |
|---|---|---|---|---|---|---|
| Poisson | 2 | **0.0** | 1.7e-06 | **3.5e-07** | 1.6e-06 | 5/5 |
| heat | 3 | **3.4e-15** | 3.7e-02 | **3.8e-06** | 1.1e-02 | 5/5 |
| wave | 4 | **1.4e-14** | 4.6e-03 | **1.1e-06** | 3.1e-03 | 5/5 |
| square (2-D) | 4 | **0.0** | 1.6e-01 | **2.8e-05** | 7.3e-02 | 5/5 |
| seam (periodic) | 3 | **1.0e-14** | 9.6e-02 | **3.6e-05** | 5.7e-05 | 5/5 |
| heat (periodic, gauge-pinned) | 4 | **1.3e-13** | 5.2e-01 | **2.4e-04** | 4.2e-01 | 5/5 |

Spectral arm on the gauge-pinned heat row (not parameter-matched to the MLP):

| arm | boundary violation (median) | interior rel-L2 (median) | beats hard | beats soft |
|---|---|---|---|---|
| spectral (`basis="spectral"`, `K=8`) | **3.3e-16** | 3.9e-03 | 0/5 | 5/5 |

The boundary column is the *falsifier* for the exactness claim, not evidence for it: the hard arm is exact by
construction, and it is reported as a max rather than a median so a single bad cell could not hide. The interior
column is the one that decides whether absorption is worth using, and it is **optimised, not proven**. The
parabolic and hyperbolic gaps are the large ones because that is where the soft arm has an initial condition
competing with the interior residual for the same gradient budget; the elliptic gap is ~4x, not ~3000x.

**What the boundary column cannot falsify.** It scores exactly the orders each condition declares. For Dirichlet
and Neumann that is the whole condition, so the column is a real test. For a **periodic** seam it is not: the
condition declares `PERIODIC_ORDERS = (0, 1, 2)` and the cage enforces those same three, so the column is graded
on its own syllabus and could never report a kink one order up. A separate probe covers that, re-declaring the
seam at the first *unmatched* order and reassembling:

| arm on a periodic row | seam jump at order 3 | as a fraction of that derivative's own scale |
|---|---|---|
| seam, hard | 6.1e+00 | 2.5% |
| seam, soft | 6.6e+00 | 2.6% |
| heat-periodic, hard | 2.6e+01 | 11.1% |
| heat-periodic, soft | 1.1e+02 | 63.9% |
| heat-periodic, spectral | **0.0** | **0.0%** |

So the honest claim for a cage is C² matching, not smoothness: `u'''` really does jump, by roughly the size of
`u'''` itself. Two things follow from the column. The hard arm is better behaved than soft *beyond* its contract
as well as inside it (11% against 64% on the gauge-pinned row), so absorption is not buying its exactness by
displacing the error just past where anyone is looking. And the **spectral arm is the only one that closes the
seam at every order**, because periodicity there is a property of the Fourier basis rather than three
constraints bolted onto a generic one -- an advantage the interior-L2 column does not show at all. Both cage
test suites pin this contract boundary in both backends, so "exact seam" cannot quietly widen into "smooth seam".

**The gauge-free seam row now wins under the shipped default** `PERIODIC_ORDERS = (0, 1, 2)`
(value, slope, and second derivative): hard wins 5/5 on interior L2 (`3.6e-05` vs `5.7e-05`), with a seam
residual at machine zero on all three matched orders. That is a flip from the earlier `(0, 1)` measurement,
where hard lost ~3x on every seed (absorbed 2, hard wins 0/5). That old row also illustrates the paragraph
above: it advertised a boundary violation of exactly `0.0`, and the same solutions graded on `(0, 1, 2)` instead
score `2.6e-01`. They had closed the seam to C¹ and no further. (The interior figure removes the mean from both
sides, since that gauge freedom belongs to the problem, not to either arm.)

**Sweep verdict (C¹ seam), and where to stop.** `benchmarks/hard_conditions_periodic_sweep.py` sweeps
`periodic_orders ∈ {(0,1), (0,1,2), (0,1,2,3)}` × `hidden ∈ {48, 96, 192}` over 3 seeds on that same gauge-free
Poisson seam, with the decision rule fixed up front: gap ~ `1/hidden` → lost degrees of freedom; `(0,1,2)`
closing the gap → C¹ seam (default should change); neither → gauge. Measured medians (hard − soft interior L2):

| orders | hidden=48 | hidden=96 | hidden=192 |
|---|---|---|---|
| `(0, 1)` | +8.7e-05 (hard loses) | +7.6e-05 | +6.1e-05 |
| `(0, 1, 2)` | **−2.9e-05** (hard wins 3/3) | **−2.2e-05** | **−1.8e-05** |
| `(0, 1, 2, 3)` | **−6.8e-05** (hard wins 3/3) | **−3.5e-05** | **−2.7e-05** |

The `(0, 1)` gap shrinks only weakly with width (log-log slope vs `1/hidden` ≈ 0.26, not ≈ 1), so this is
**not** a pure degrees-of-freedom cost. Raising the matched orders to `(0, 1, 2)` flips the comparison: hard
wins on every seed. That implicates the C¹ seam under a second-order operator (quadratic switching leaves
`u''` discontinuous across it).

`(0, 1, 2, 3)` answers a different question -- *how far to go* -- under its own rule, also fixed before the run:
raise the default again only if the fourth matched order buys more than the third did. That has to be read on
the hard arm alone, because the gap column above conflates it with the soft arm getting worse as rows are added.
The third order gains **3.34x**, the fourth **1.30x**:

| orders | hard interior rel-L2 (hidden=96) | gain over previous |
|---|---|---|
| `(0, 1)` | 1.21e-04 | -- |
| `(0, 1, 2)` | 3.61e-05 | 3.34x |
| `(0, 1, 2, 3)` | 2.79e-05 | 1.30x |

**The default stays `(0, 1, 2)`.** A smooth manufactured solution matches every derivative, so it rewards extra
orders indefinitely and "better here" is not grounds to ship them; diminishing returns are the signal to stop.
The default also has to hold on problems that are *not* smooth, and the periodic-emit measurement already showed
Burgers losing interior accuracy when its seam is enforced at all -- over-smoothing a seam near a steep gradient
is exactly the failure a higher default would deepen. Artifact:
`docs/benchmarks/hard_conditions_periodic_sweep.json`; the main solver table above was regenerated under the
shipped default.

**The gauge-pinned heat row keeps the story.** With IC `u(x,0) = sin(2πx)` and exact
`exp(-α(2π)² t) sin(2πx)`, soft's additive freedom buys nothing: hard wins 5/5 (interior `2.4e-04` vs
`4.2e-01`). The spectral arm beats soft on every seed (`3.9e-03`) and is the one arm whose seam closes at every
order rather than at three of them, but it still does not beat the MLP cage on interior accuracy at this budget
-- free periodicity is not free accuracy when the Fourier time-head is under-resolved relative to the one-layer
MLP.

Absorption is partial and opt-in. `hard_conditions="auto"` absorbs what the planner can certify and leaves the
rest soft, reporting a reason per declined condition; the default `"none"` reproduces the previous solve bit for
bit. Conditions on any number of axes are in scope -- the square row is two constrained spatial axes, whose
corner terms the recursion generates on its own -- and a condition the planner cannot certify is declined, which
is the same answer the solver gives today.

### DeepONet operator learning (closed-form residual vs FD)

CPU smoke, regenerable from the scripts below. Physics weight `λ` and Burgers
time spacing / KS stencil step `h` are read off
[`operator_residual_calibration.json`](benchmarks/operator_residual_calibration.json)
(short-budget curves; Burgers bake-off uses `λ=0.1`, `n_times=11`; KS uses
`h=0.01` and a train-clearing `λ=0.1` override — calibration's max-gap `λ=1`
failed the train-rel-L2 guard on some seeds).

#### Burgers (1st-order-in-time FD vs closed form)

[`benchmarks/operator_deeponet.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/operator_deeponet.py)
→ [`operator_deeponet.json`](benchmarks/operator_deeponet.json). A DeepONet learns
the periodic viscous-Burgers solution operator `u(·,0) ↦ u(·,t)` on a space-time
slab; ground truth is the spectral MOL reference. Three arms share architecture
(~23k params), seed and **step budget** (1500 Adam steps; FD `dt` is derived from
the slab's time column). Convergence guard: train rel-L2 ≤ 0.35.

| arm | held-out rel-L2 (median / IQR / 8 seeds) | median wall (s) | notes |
|---|---|---|---|
| A — data-only Adam | **0.259** / 0.069 | 26 | beats B and C on 7/8 seeds |
| B — physics-informed, `u_t` by finite difference | 0.269 / 0.067 | 374 | — |
| C — physics-informed, closed-form trunk jet | 0.269 / 0.068 | 423 | beats B on 2/8 seeds |

**Verdict (decision rule fixed before the run).** Honest negative on Burgers:
closed-form residual (C) does **not** beat FD residual (B) on median held-out
rel-L2 under this budget (C wins the head-to-head on only 2/8 seeds). That is
expected once the FD error in `u_t` at `dt=0.05` sits near **1e-8** — far below
anything the training loss can feel — so B and C are near-identical by
construction. Data-only Adam (A) still beats both physics arms on 7/8 seeds.
The 4th-order claim is carried by the KS bake-off below, not by Burgers.

#### KS (4th-order FD `u_xxxx` vs closed form)

[`benchmarks/operator_ks_bakeoff.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/operator_ks_bakeoff.py)
→ [`operator_ks_bakeoff.json`](benchmarks/operator_ks_bakeoff.json). Same three
arms on periodic Kuramoto–Sivashinsky (`jet_order=4`; arm B uses
`ks_residual_loss_fd` with a 5-point stencil at the calibrated `h=0.01`). Slab
shortened vs the house stiff-test (`n_grid=64`, `t_final=0.5`, `amplitude=0.3`)
so the DeepONet clears train-rel-L2 ≤ 0.35 at `lr=3e-3` / 2000 steps on CPU.

| arm | held-out rel-L2 (median / IQR / 8 seeds) | median wall (s) | notes |
|---|---|---|---|
| A — data-only Adam | 0.205 / 0.080 | 20 | beats both on 3/8 seeds |
| B — physics-informed, `u_xxxx` by 5-point FD | 0.220 / 0.095 | 1643 | — |
| C — physics-informed, closed-form trunk jet | **0.198** / 0.126 | 1404 | beats B on **5/8** seeds |

**Verdict (decision rule fixed before the run):** claim the 4th-order FD-floor
training win only if C beats B on median **and** on ≥6/8 seeds. Measured: C
beats B on median (0.198 vs 0.220) but only on **5/8** seeds — seed-fragile.
Do **not** advertise a seed-stable training win; keep the structural claims.
Trained-model mechanism diagnostic `|FD u_xxxx − closed-form u_xxxx|` median ≈
**4e-7** (arm B), consistent with a polluted biharmonic term inside the residual.

**Structural claims, independent of the training outcome.** Query-coordinate
derivatives of `G(u)` are closed form through order 4 (machine-epsilon vs nested
autograd; see [`operator_fd_floor.json`](benchmarks/operator_fd_floor.json):
5-point `u_xxxx` floors at **4.5e-6** while the closed form sits at **2e-17**).
A 4th-order residual costs exactly one trunk jet. One trunk jet is shared across
a batch of input functions
([`operator_shared_grid.json`](benchmarks/operator_shared_grid.json): ~24× at
`F=32`). A residual enclosure over a sensor / coefficient family seals as a
sound interval certificate (not a solution-error bound).

**FNO baseline** ([`operator_fno_vs_deeponet.json`](benchmarks/operator_fno_vs_deeponet.json)):
matched parameter count (FNO `modes=8 / width=20 / layers=3` → ~20.5k vs DeepONet
~23k) and step budget on the same Burgers target. FNO wins on its natural
final-time grid map (median hold rel-L2 **0.036** vs DeepONet space-time
**~0.26**), and supplies **no** off-grid `u_t` or `u_xxxx` at all — that
asymmetry is the structural point of the comparison.

## Structured DP (omnibias-struct)

`omnibias-struct` runs one soft-DP substrate (`logsumexp_beta` relaxation, exactly
differentiated by the `delta -> 0` softplus/softmax jet tower) across Viterbi,
shortest-path, CTC, soft-DTW, alignment (global / local Smith-Waterman / affine-gap Gotoh),
planning, MAS, and structured attention -- and, on the **semiring / hypergraph driver**, the
tree / grammar families **CKY inside-outside parsing**, **Eisner projective dependency**,
and the **matrix-tree** non-projective marginals, plus the **distribution operators** (path
entropy, exact sampling, exact k-best). A **data-driven probe harness**
(`packages/omnibias-struct/tests/_harness.py`) turns each soft-DP *claim* into a measured
number + a boolean verdict; the same probes power the CPU-tiny regression subsets
(`tests/test_refinement_probes.py`, `tests/test_refinement_families.py`) and a multi-seed
sweep (`struct_refinement/sweep.py`, in the separate `omnibias_experiments` project) whose aggregate is committed as
[`struct_refinement.json`](benchmarks/struct_refinement.json). The DP work is CPU-bound and
float64, so the sweep runs identically on a CPU-dev host or a GPU-cluster job.

The committed run below is a **24-seed sweep** -- the base sequence families {Viterbi,
shortest-path, CTC} x up to four sizes (to `T = 96` / `n = 20`) *plus* the driver families
{CKY parse, Eisner, matrix-tree, local + affine alignment} and the distribution operators
-- across nine temperatures `beta in {1 ... 1e6}` x {torch, jax} = **21,048 probes, all
green** (commodity CPU node, ~1 h 48 m):

| Probe (axis) | What it measures | Worst over the sweep |
|---|---|---|
| `oracle_agreement` (`beta -> inf`) | hard DP == brute-force enumeration (base + all driver families) | `<= 8.9e-16` (bit-exact; family derivation count == tree count) |
| `path_count` | log-space count vs `log` of the exact integer count | `<= 2.2e-16` |
| `gap_tightness` (`beta -> inf`) | realised `\|V_beta - V*\|` vs the `log(N)/beta` bound | ratio in `[0, 0.80]`, always sound |
| `parity` | torch vs jax soft value (base + driver families) | `<= 5.7e-14` (float64, up to `beta = 1e6`) |
| `marginals_vs_autodiff` (`delta -> 0`) | closed-form forward-backward / inside-outside vs backend autodiff | `<= 3.4e-7` at the `beta = 1e6`, `T = 96` corner; `<= 1.3e-9` for the tree families |
| `entropy` (`delta -> 0`) | path entropy identity `H = beta(V_beta - E[score])` vs enumeration | `<= 1.4e-7` (MC cross-check `mc_err ~ 3e-2`) |
| `sampling` / `kbest` | exact-sampler empirical marginals; exact k-best vs enumerate-and-sort | MC marginals `~ 1e-2` (20 k draws); k-best score diff `== 0` (bit-exact hard decode) |
| `beta_stability` | value / marginals stay finite + normalised | finite, `sum -> 1` for all `beta` |

The matrix-tree family is the **exact Kirchhoff-determinant** partition (not an `lse_beta`
relaxation): torch<->jax parity `<= 8.9e-16` and marginals `== L^{-1}` autodiff to `3.3e-16`,
with the `beta -> inf` gap sandwiched against the Chu-Liu/Edmonds maximum arborescence.

### Flaws found -> fixed (before -> after)

The harness was written as a flaw-finder first; every issue it surfaced became a regression
test. All numbers are CPU-dev, float64.

| Flaw surfaced | Before | After |
|---|---|---|
| CTC had no closed-form marginals and no hard traceback | `soft_ctc_marginals` absent; `ctc_best` returned a float only | `soft_ctc_marginals` (+ torch/jax twins) matches autodiff to `<= 1e-9`; `ctc_best` returns the aligned path |
| Path count lost precision / crashed | `math.log(num_paths)` overflows for `N > 2^53`; `num_paths == 0` raised | log-space `paths.py` count is overflow-free (exact to `2.2e-16`, and finite at `10^400` paths) and gap-safe when `N in {0, 1}` |
| Gap bound was global-only | only `log(N)/beta`, loose on DAG/CTC | per-step `stepwise_gap_bound`; `certify_soft_dp` takes the tighter of global/stepwise |
| Large-`beta` untested / float32 sentinel | validated only to `beta = 256`; `-1e30` sentinel unguarded | green to `beta = 1e6` in float64; documented working-precision envelope |
| Marginals vs autodiff assumed `1e-9` everywhere | fixed `1e-9` threshold fails at `beta = 1e6` on longer chains | verdict uses the honest float64 envelope `~ eps * beta * L` (`~3.4e-7` at `beta = 1e6`, `T = 96`); strict `1e-9` retained at moderate `beta` |
| Matrix-tree `det` / `L^{-1}` went singular at high `beta` | raw `standard_normal` arcs whose greedy argmax is a *cycle* drive the exp-Laplacian minor singular by `beta ~ 16` (an intrinsic limit of the determinant route, crashing the sweep) | probes use well-separated tree-optimum arcs (`_make_arc_mtt`), keeping the minor conditioned (`cond ~ 4`) and green across the whole `beta` ladder; the degenerate limit is pinned by `test_matrix_tree_singular_at_high_beta_for_cyclic_argmax` |

The two axes are kept labelled and never conflated: `beta -> inf` is the *relaxation*
(soft-DP -> hard DP), and `delta -> 0` is the *bias-collapse derivative tower* that
differentiates `logsumexp_beta` exactly. The `gap_tightness` sandwich certifies the
relaxation error (temperature axis), **not** model correctness.

## Tabular learning (omnibias-tab vs LightGBM)

`omnibias-tab` trains **soft oblique decision trees** -- a split `1[w.x > t]` becomes a
gate `sigmoid(beta (w.x - t))` annealed `beta -> inf` toward a hard split (the feasibility
/ temperature sense of collapse) -- with an exact second-order trainer and a Newton-boosting
driver. The empirical-validation gate is *best-in-class*: **match or beat LightGBM** on a
fair, standardized, stratified train/test split, judged "not worse" iff `tab`'s mean
held-out metric is within the baseline's own across-seed noise (or better). The head-to-head
harness (`omnibias.tab.bench`) and both reproducers -- the CPU-smoke
[`docs/examples/tab_validate.py`](https://github.com/derivon-ai/omnibias/tree/main/docs/examples/tab_validate.py)
and the cluster sweep `packages/omnibias-tab/bench/sweep.py` -- are committed; the metric is
reported so **higher is better** (accuracy for classification, `-RMSE` for regression).

**Full-data, boosted `tab` (depth-2 oblique stumps, 60 stages, no per-dataset tuning) vs
LightGBM (200 trees), mean ± std over `K = 8` seeds; `scikit-learn` built-in datasets,
float64, single-threaded.** Result: `tab` is **not worse than LightGBM on 4/4 datasets and
strictly better on 3/4** (`digits` is a statistical tie -- `0.9742` vs `0.9747`, inside the
baseline's own `±0.0059` seed noise).

| Dataset | Task | Metric (higher = better) | omnibias-tab | LightGBM | not-worse |
|---|---|---|---|---|---|
| breast_cancer | binary | accuracy | **0.9738 ± 0.0130** | 0.9677 ± 0.0099 | yes (win) |
| wine | multiclass | accuracy | **0.9806 ± 0.0133** | 0.9778 ± 0.0222 | yes (win) |
| digits | multiclass | accuracy | 0.9742 ± 0.0038 | 0.9747 ± 0.0059 | yes (tie, within noise) |
| diabetes | regression | RMSE (lower) | **58.93 ± 1.36** | 61.67 ± 2.46 | yes (win) |

The deterministic CI smoke (`tab_validate.py`) additionally checks, on every push:
bit-identical torch<->jax forward (`<= 1e-9`, float64); a *proved* per-feature monotone
constraint plus a sound output enclosure and a certified soft->hard rounding gap; and the
**exact second-order trainer strictly beating a tuned Adam baseline** on held-out data at
a matched step budget. `tab_as_layer.py` checks that `SoftTreeEnsemble` /
`ArrangementBoosted` compose with an encoder and that joint Adam updates both sides.

!!! note
    The four datasets above are the offline, network-free suite (reproducible anywhere). The
    wider network datasets (`california_housing`, and the OpenML `adult` / `higgs` subsets)
    are fetched only where a node has outbound network; the harness *skips them cleanly* when
    a fetch is refused (they are absent above because this run had no network), and the cluster
    sweep (`packages/omnibias-tab/bench/sweep.py`, artifacts to `artifacts/`) transcribes them
    here when available. Per the empirical-validation discipline: if a dataset stays behind
    after refinement it is *reported and the claim scoped*, never hidden by loosening a test
    or tuning to one seed.

The standalone value holds even at accuracy parity: `tab` trees are **differentiable**
(composable / fine-tunable; GBM trees are not), **exactly second-order-trained** (splits
included, not just leaf Newton steps), **certified** (sound output / Lipschitz /
monotonicity / rounding-gap bounds), and **bit-identical across torch / jax**.

## Reproducing

CPU-smoke results:

```bash
pip install -e "packages/omnibias-core" "packages/omnibias-torch[test]" "packages/omnibias-jax[test]"
python -m pytest tests/ -q
python docs/examples/quickstart_jax.py
```

Tabular head-to-head (CPU smoke + the full cluster sweep):

```bash
pip install -e "packages/omnibias-tab[torch,jax,verify,gbm]"
python docs/examples/tab_validate.py                       # deterministic CPU smoke (CI)
python docs/examples/tab_as_layer.py                       # encoder + tab head plugin (CI)
python docs/examples/tab_as_layer_jax.py                   # JAX encoder + arrangement/tree kernels (CI)
python packages/omnibias-tab/bench/sweep.py --seeds 8      # full multi-seed suite (cluster)
```

GPU headline results require GPU-class hardware; run the corresponding
training script on a GPU job and transcribe the printed metrics into the
table above.
