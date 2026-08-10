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
- **Stretch still unearned:** best measured dense Wang residual ~`1.2e-1`
  (gate `1e-13`). Do not claim progress past the measured floor.

## Known stretch blocker (audit)

Hilbert × dictionary catch-22:

- Spectral / PV Hilbert alone err at **O(1e-1)** vs exact `H[Q]=-P`.
- `hardy_corrected_pv` + high `proj_defect_weight` pulls Ω into a Hardy span
  that itself floors near **~1e-1** under Martens–Grosse (richer JAX dict
  probes ~0.07–0.09).
- More MG / origin-only collocation on the same depth-2 net does **not** clear
  1e-13. Next bets: richer Hardy dictionary **or** high-accuracy whole-line
  Hilbert for free neural Ω (prefer GPU cluster throughput).

## Optimizer doctrine

- **Phase 0 (reproduce):** Martens–Grosse Gauss–Newton (exact JVP) on compactified
  neural Ω; train Hilbert **`hardy_corrected_pv`** (exact `H` on Hardy projection
  + autograd-safe PV on the remainder). Gate scores
  `max(residual, projection_defect)`. Spectral FFT alone floors ~`1e-1`. Adam
  warmup allowed only on this labeled arm (cold start); escalate uses Adam=0.
- Multistage `optimizer="gauss_newton"` is a **corr-matching proxy**
  (`gauss_newton_corr_proxy`), not linearized Wang GN — do not claim otherwise.
- **Laptop GPU (e.g. T1200 4GB):** float64 CUDA works; keep `hidden≤80`,
  `n_grid≤201`, one CUDA job at a time; prefer vectorized Hardy fields.
- **Rung-1 earn (after stretch):** `CubicGaussNewton`, QR `GaussNewton`,
  Martens–Grosse on Hardy-Ω (`train_gn` / `martens_grosse_gauss_newton_minimize`),
  mpmath polish, multistage. **Adam forbidden** for Rung-1 earn. Train Hilbert
  must be Hardy (`hardy_projection` / exact), matching Rung/CAP.

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
