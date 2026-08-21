---
name: omnibias-dev-deepmind-campaign
description: >-
  Run the DeepMind-style autonomous unstable-singularity campaign in omnibias —
  phase-0 neural CCF reproduction (Martens–Grosse, hardy_corrected_pv) to
  1e-13, then Hardy CCF Rung-1/2, IPM/Boussinesq, Phase 5. Use when iterating
  campaign ticks, closing residual gates, or wiring /loop autonomy.
---

# DeepMind-style singularity campaign (maintainer)

## Honesty

- Claim only earned absolute gates. Never Clay / continuum Navier–Stokes.
- `navier_stokes_proof_claim=False` always on campaign artifacts.
- Phase 5 blocked until `whole_line_certified=True`.
- **Reproduce first:** neural line CCF (DeepMind recipe) before Hardy dictionary / CAP.
- **Stretch still unearned:** official-path dense Wang residual is
  `7.840e-3` (even compact Gaussians at the origin and
  residual peak `|y|≈1.50`, epigraph L∞ then one refine;
  `peak_bump_best.npz`; `HΩ(0)≈+1.007`; residual peak
  `|y|≈0.10`). Parent tanh-even readout was `7.847e-3`
  (`tanh_even_best.npz`). Parent even Fourier \(k=6..12\) was
  `7.876e-3` (`fourier_hi_best.npz`). Parent even Padé-on-`q`
  was `7.884e-3` (`pade_q_best.npz`). Parent even
  Legendre-on-`q` was `7.885e-3` (`legendre_q_best.npz`).
  Parent even Chebyshev-on-`q` was `7.889e-3` (`cheb_q_best.npz`).
  Parent second even-`q` MSNN was `7.938e-3` (`msnn_stage3_best.npz`).
  Parent first even-`q` MSNN was `7.943e-3` (`msnn_even_q_best.npz`).
  Parent even Fourier stage-2 was `7.991e-3`
  (`even_fourier_stage2_best.npz`). Parent signed
  PirateNet residual correction was `8.063e-3`
  (`pirate_hat_best.npz`). The pad stack under that correction
  is 11-node cubic Hermite plus frozen dual even
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
  then even compact Gaussians at the origin and `|y|≈1.50`.
  An identity-init SiLU MLP readout on the same even
  `(q, ξ, bump)` features did not promote (LP pred
  earn `~1.4e-7`; line search raised L∞). Unfreezing the
  tanh hidden as additive `ΔW`/`Δb` (frozen readout) did
  not promote (LP pred earn `~1.74e-6`; line search raised
  L∞). A multiplicative even Fourier on
  \(\xi=y^2/(1+y^2)\) (`1+g` and paper exp-mult) did not
  promote (LP pred earn `~8.2e-5`; line search raised L∞).
  Identity-init GELU on \((\rho, E, \rho-q)\) with
  \(\rho=(1+y^2)^{-1/2}\) did not promote (LP pred earn
  `~3.4e-7`); paper L2 moved `HΩ(0)` toward `+1.30` but
  raised 1601-pt L∞. Identity-init paper MSNN on
  \(\eta=[(2/\pi)\arctan y]^2\) did not promote (LP pred
  earn `~1.87e-5`; every improving L∞ step walked the
  peak to `|y|≈38` and was rejected). Identity-init Mish
  on \(\zeta=\log(1+y^2)/(1+\log(1+y^2))\) did not
  promote (tail-capped LP pred earn `~1.59e-6`; line
  search raised L∞). Even \(\operatorname{sinc}(y/s)\)
  realized a true L∞ earn of `~7.0e-7` (peak `|y|≈2.60`)
  after tail-capped LP + one refine, below the `1e-6`
  promote gate. Even cylindrical \(J_0(y/s)\) did not
  promote (LP pred earn `~7.5e-7`; improving steps
  walked the peak to `|y|≈38`). A residual-shaped even
  lift \(r/y\) and \(r y/(1+y^2)\) realized only
  `~3e-11`. Unfreezing the PirateNet last layer
  (`ΔWout`/`Δbout`) did not promote (LP pred earn
  `~7.4e-6`; line search raised L∞). Even Hermite–Gauss
  \(\mathrm{He}_{2k}(y/s)\,e^{-y^2/2s^2}\) did not promote
  (LP pred earn `~3.6e-5`; line search raised L∞; this
  was not an unfreeze of the frozen Hermite pad).
  Unfreezing stage-3 MSNN Fourier frequencies `ΔB`
  did not promote (LP pred earn `~3.4e-6`; line search
  raised L∞). Scored on 1601-pt L∞;
  `HΩ(0)≈+1.007`; residual peak `|y|≈0.10`; gate `1e-13`).
  The tanh-even parent was `7.847e-3`; the Fourier-\(k=6..12\) parent was `7.876e-3`; the Padé-on-`q` parent was `7.884e-3`; the Legendre-on-`q` parent was `7.885e-3`; the Chebyshev-on-`q` parent was `7.889e-3`; the stage-3 MSNN parent was `7.938e-3`; the first-MSNN parent was `7.943e-3`; the even-Fourier parent was `7.991e-3`; the PirateNet parent was `8.063e-3`; the erf-sinh parent
  was `8.068e-3`; the circular-Planck
  parent was `8.227e-3`; the alg7 parent was
  `8.232e-3`; the explag1 parent was
  `8.234e-3`; the alg15-Gegenbauer parent was
  `8.278e-3`; the half-stereo parent was
  `8.279e-3`; the alg5 parent was `8.294e-3`;
  the Laguerre parent was
  `8.297e-3`; the alg6-Jacobi parent was
  `8.299e-3`; the log1p-Gegenbauer parent was
  `8.309e-3`; the alg2-U parent was `8.356e-3`;
  the alg3-Legendre parent was `8.365e-3`; the softsign parent
  was `8.37e-3`;
  asinh was `8.38e-3`; alg4 was `8.39e-3`; erf was `8.43e-3`;
  tanh-sinh was `8.44e-3`; Boyd was `8.46e-3`; tanh was
  `8.48e-3`; `s=2.8` was `8.50e-3`; `s=1.8` was `8.53e-3`;
  `s=0.30` was `8.56e-3`; L∞ parent was `1.008e-2`; Laplace was
  `1.24e-2`; rat4 was `1.25e-2`; C¹ was `1.28e-2`. Peak-weighted
  `p=12` was `1.80e-2`; unweighted Hermite was `1.90e-2`. Parent
  dual-scale arctan-even plus Gaussian pads was `4.44e-2`.
  Ker / abs-shift were `8.35e-2` / `7.49e-2`. Hop-218 U-refine
  `1.9e-5` uses the lineage `fields()` operator (MO velocity double-count);
  it is not a valid Wang residual on `U'=HΩ`.

## Known stretch blocker (audit)

Ranked:

1. **High-accuracy whole-line H for free neural Ω** (fork
   `enrich_dict_or_free_omega_hilbert`). Library:
   `omnibias.pinn.{jax,torch}.hilbert_line.hilbert_wholeline_hp` and
   `train_hilbert="wholeline_hp"`. Planted `H[Q]=-P` sits near `1e-14`
   (wide) / `1e-13` (narrow `a=0.25`). Campaign GL96+raw-`u` sat at
   `1e-3`–`1e-1`. On hop 218 the remainder Hilbert disagreed by
   `3.14e-4` (larger than the `1.9e-5` lineage residual). On the
   official Wang warm net, hp and `hardy_corrected_pv` agree at `0.120`
   — that net is a ghost (`raw Ω ~ 1e-6`). Stretch is still unearned.
2. **A dictionary that can actually represent the profile** — not more N
   on the same `{P,Q}` family. Dictionary enrichment was tried (leftover
   products, new-scale-only, recovery GN on consistent-U). Closed.
3. Do not confuse gates. DeepMind “Earth to centimeters” is a float64
   training residual. `1e-13` here is an absolute CCF stretch gate.
   Float64 can store it; the Hilbert you were calling could not.
4. 03-10 jet–Padé does not clear stretch (diagnostic only).

Official GPU CompactifiedOmegaOMBU is closed (ghosted). Prefer JAX CPU
or torch CPU `wholeline_hp`. One heavy job at a time.

## Which tool (do not roll your own)

| Task | Use | Do not use |
|---|---|---|
| Hardy-Ω Wang residual | `vorticity_residual_samples` / `dense_vorticity_residual` / `hardy_omega_profile` (closed-form `H`, `U`) | trapezoid `U` of `H`; `jnp.gradient` `Ω_y` |
| Free-Ω Wang residual | `free_omega_vorticity_residual` + analytic `Ω_y`; Hilbert `hilbert_wholeline_hp`; `U` via `integrate_velocity_from_hilbert` | `OperatorBlock` `integral` (activation windows, not `∫HΩ`); `omnibias.fields` grad (not Hilbert); FD `Ω_y` |
| Phase-0 train | `CompactifiedOmegaOMBU` + `train_hilbert="wholeline_hp"` + Martens–Grosse | official GPU CompactifiedOmegaOMBU; hop-218 lineage `fields()` |
| CCF stretch L∞ earn | epigraph L∞ LP / `linearized_linf_direction` on a live Jacobian; promote on 1601-pt L∞ | L2 GN / `martens_grosse` / `wang_linearized_gn` / Adam as the L∞ metric (measured: they raise or stall `max|r|`) |
| Signed-hat search | `ccf_hat_homotopy` (nodal init, frozen-velocity Picard, `homotopy_gauss_newton_minimize`); signed PirateNet `pirate_hat` (identity-init + PI-init / hat-L2, no softplus) | positive-only / softplus hat (cannot flip `HΩ(0)`); official GPU CompactifiedOmegaOMBU; installing jaxpi / fluid-singularities / Unstable-Singularity-Detector as a CCF `1e-13` solver; Rung-1 from this free-Ω driver |
| Hardy `U` | `hardy_omega_profile` / `conjugate.hardy_omega_velocity_atom` | trap of numerical `H` on a Hardy span |
| CAP | `certified_ccf_hardy_wholeline` **after** stretch | not a root finder |

`Ω` must be classically `C^1` for a `max|r|` that contains `Ω_y`. The official envelope `y(1+y²)^{-(α+1)/2} hat` is `C^∞` for smooth `hat`. Profiles with `Ω ∼ |y|^{1-α} sgn(y)` near 0 (`erf` hole) have **unbounded** `Ω_y` and are not this path.

Score free-Ω residuals on `|y| < y_trunc`. The hp tail is not exact on the truncation nodes.

## Architecture vs optimizer (Wang et al., not Medium)

The paper stack is: compactified even \(q\), envelope \(\Omega=y\cdot E\cdot\mathrm{hat}\),
small \(\tanh\) MLP, **exp-adjacent last layer**, gradient-normalized residual,
MSNN stage-2 Fourier net, full-matrix Gauss–Newton (kfac-jax + Martens–Grosse
closed-form LR). Omnibias matches every inductive bias and has a *stronger*
Jacobian (exact JVP, not rank-1 EMA). Helpers:
`deepmind_paper_architecture_config`, `deepmind_signed_hat_config` (no
softplus; official `exp_core=True` ghosted), `deepmind_multistage_config`
+ `stage2_even_hat` (even \(q\); raw-`y` Fourier is not CCF-faithful).
PirateNet is jaxpi, not this paper. Reuse the paper stack on a stage-1
  that already represents the champ (even `q`, envelope, small tanh
  Fourier MSNN, identity readout, optional exp-adjacent *correction*,
  gradient-normalized loss, eq. 19 / Martens–Grosse). Stretch is still
  unearned because stage-1 never reached the \(10^{-8}\) basin MSNN
  needs; the current champ is even compact Gaussians on the
  frozen pad+pirate+Fourier+MSNN+poly+Padé+Fourier-hi+tanh-even
  stack at \(7.840\times 10^{-3}\).
  Measured again on that basin (2026-08-21): paper L2 / grad-norm
  raised 1601-pt L∞; additive epigraph L∞ is the only earn path.
  Keep L∞ for this stretch
  metric until a paper loss actually beats it. `martens_grosse` /
`wang_linearized_gn` / grad-norm stay the paper trainers for a
future \(10^{-8}\) stage-1; Adam stays the smoke heuristic.

## Optimizer doctrine

- **Phase 0 (reproduce):** Martens–Grosse Gauss–Newton (exact JVP) on compactified
  neural Ω. Default train Hilbert **`hardy_corrected_pv`** (exact `H` on Hardy
  projection + PV on the remainder) still floors the official Wang net.
  The open free-Ω operator is **`train_hilbert="wholeline_hp"`**. Gate scores
  `max(residual, projection_defect)` on Hardy modes. Spectral FFT alone floors
  ~`1e-1`. Adam warmup allowed only on the labeled reproduce arm (cold start);
  escalate uses Adam=0.
- Multistage `optimizer="gauss_newton"` on a numpy residual is a
  **corr-matching proxy** (`gauss_newton_corr_proxy`). The eq. 19
  residual-vector path is JAX `optimizer="martens_grosse"` /
  torch `optimizer="wang_linearized_gn"` (requires a torch-graph
  residual). Neither forges stretch.
- **Laptop GPU (e.g. T1200 4GB):** float64 CUDA works; keep `hidden≤80`,
  `n_grid≤201`, one CUDA job at a time; prefer vectorized Hardy fields.
- **Rung-1 earn (after stretch):** `CubicGaussNewton`, QR `GaussNewton`,
  Martens–Grosse on Hardy-Ω (`train_gn` / `martens_grosse_gauss_newton_minimize`),
  mpmath polish, multistage. **Adam forbidden** for Rung-1 earn. Train Hilbert
  must be Hardy (`hardy_projection` / exact), matching Rung/CAP.
- **Gauge-hard homotopy (Phase 0, free Ω):**
  `homotopy_gauss_newton_minimize` on `L + t Quad` with a *signed*
  official-envelope hat (`ccf_hat_homotopy`), nodal init, and
  frozen-velocity Picard. Hard gauge belongs **inside** the residual JVP.
  `champ_barrier_residual` is the L^∞ / no-walk successor of
  `peak_weighted_residual` (the `p`-ladder stalled at `p=14`).
  A direct L∞ line search on a frozen-champ correction beat
  barrier-GN when the Jacobian of the new family was dead.
  When the Jacobian is live but L2 Newton raises `max|r|`, use
  `linearized_linf_direction` (or its epigraph-LP twin) and
  promote only on 1601-pt L∞. Measured three times (linear even Fourier,
  identity-init even-`q` MSNN, and even Chebyshev-on-`q`, 2026-08-21): ranking by
  1601-pt L∞ was `linf_epigraph` first; paper L2 / grad-norm /
  exp-adjacent multiplicative / IRLS all raised `max|r|`.
  Grad-norm is the follow-up *training* loss for high-gradient
  λ-search, not a substitute for this L∞ gate at the `8e-3` basin.
  Reject steps that move the peak
  to far-field nodes for a sub-`1e-6` gain.
  Positive-only hats cannot flip `HΩ(0)`. Does not skip stretch. Does
  not start Rung-1.

## Local autonomy hygiene

- Kill hung trial scripts and eternal `/loop` tick shells before starting new
  CUDA work (one job at a time).
- Warm lineage: `warm_net_reproduce.pt` + `warm_best_residual.json` are canonical;
  escalate refreshes `warm_net_ab.pt` from reproduce when the reproduce floor is
  better. Save warm **only on residual improve**.
- Prefer `$OMNIBIAS_SUBMIT` / GPU cluster for Fourier / capacity / Hilbert sweeps.

## Commands

```bash
# Phase-0 neural reproduction smoke
uv run python benchmarks/reproduce_deepmind_ccf.py --write-docs

# Escalate toward 1e-13 (prefer GPU / $OMNIBIAS_SUBMIT for --full)
uv run python benchmarks/reproduce_deepmind_ccf.py --full --escalate

# One autonomy tick (phase0 until stretch, then Hardy ladder)
uv run python benchmarks/deepmind_campaign_tick.py

# Hardy acceptance (after stretch)
uv run python benchmarks/ccf_hardy_rung_acceptance.py
uv run python benchmarks/ccf_hardy_rung_acceptance.py --full
```

## Gate order

0. Phase 0: dense neural Wang residual ≤ `1e-13` (`CCF_STRETCH_RESIDUAL_GATE`)
1. Rung-1: `|λ−0.6057|≤5e-5` and dense Hardy Wang residual `≤1e-11` (anti-ghost)
2. Rung-2: `whole_line_certified` (residual_sup + both NK)
3. IPM / Boussinesq absolute gates
4. Phase 5a partition → 5b tab router → 5c logic obligation planner

Never weaken `1e-13` or `1e-11`. Never forge `whole_line_certified`.
