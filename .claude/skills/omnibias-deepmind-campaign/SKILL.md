---
name: omnibias-deepmind-campaign
description: Run the DeepMind-style autonomous unstable-singularity campaign in omnibias — phase-0 neural CCF reproduction (Martens–Grosse, wholeline_hp) to 1e-13, then Hardy CCF Rung-1/2, IPM/Boussinesq, Phase 5. Use when iterating campaign ticks, closing residual gates, or wiring loop autonomy.
---

# Unstable-singularity campaign

Phase-0 reproduces the neural line CCF (Martens–Grosse, `wholeline_hp`) to
the residual gate, then Hardy CCF rungs, IPM/Boussinesq, and Phase 5 climb
on that scaffold. Closed-form jets and certified remainders are the
instruments; nested AD is the bottleneck the campaign is designed to beat.

## Why nested AD fails

Self-similar CAP residuals at the orders this campaign needs rebuild
nested-AD graphs until the line search dies. Without an exact jet, Padé /
Hermite / Chebyshev pads are floating fits. Without a sound remainder, a
champion residual is a float, not a gate.

## What only this tower unlocks

Neural CCF on the closed-form tower; Hardy dictionary pads; conjugate-sweep
and Padé-profile smokes; a path from a reproducible residual gate to a
certified whole-line profile. Reproduce first (neural line CCF) before Hardy
dictionary / CAP. Artifact provenance belongs in the smoke JSON: committed
Hardy floors vs uncommitted scratch champions.

Lambda-digit gates: with default `train_lam=False` and `funnel_updates=0`,
line discovery keeps `lam_init`; the residual gate is what establishes a
reproduction result.

## Use

- Campaign skill + `.cursor/rules/omnibias.md` (CCF / frontier).
- Smokes: `docs/benchmarks/reproduce_deepmind_ccf_smoke.json`,
  `ccf_pade_profile_smoke.json`, `ccf_line_smoke.json`,
  `ccf_conjugate_sweep_smoke.json`.
- Cookbooks: `docs/cookbook/ccf-singularity.md`.
- Pair with `omnibias-frontier`, `omnibias-pinn`, `omnibias-ferminet`,
  `omnibias-verified-primitive`.

Phase 5 reads `whole_line_certified`. Heavy regeneration follows the
workspace compute rule; artifacts under `$OMNIBIAS_SCRATCH`.

## Extend

Ticks close residual gates, then remainder enclosures, then Lean on the
finite rational core. Compose with `omnibias-empirical-validation` for
`gates` JSON. No new package — campaign code lives in existing PINN /
verified / ferminet modules.

## Next invention

A reproducible neural-line residual that clears the phase-0 gate, then a
Hardy pad whose committed smoke beats the current floor with a residual
enclosure on the same profile.

## Bakeoffs

`docs/benchmarks/reproduce_deepmind_ccf_smoke.json`,
`ccf_pade_profile_smoke.json`.

## Further references

- `docs/cookbook/ccf-singularity.md`
- `docs/frontier-ledger.md`
