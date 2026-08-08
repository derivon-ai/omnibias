---
name: omnibias-dev-pinn-research
description: Ambitious, falsifiable PINN research inside omnibias-pinn -- separate structural impossibility from absent implementation, turn every caveat into an acceptance test, iterate after a losing baseline, and claim success only when the gate passes. Use when closing PINN capability gaps (causality, geometry, operators, spectral bias), designing benchmarks, or deciding whether a limitation is real. For contributors modifying omnibias itself, not for consumers using it.
---

# PINN research doctrine (ambitious + claim-disciplined)

Closed-form activation derivatives and exact hard cages are *leverage*, not a
proof that non-convex training, finite capacity, identifiability, or
PDE-dependent stability vanish. The job is decisive closure on **named problem
families**, with machine-checkable gates -- never a universal "PINNs are solved"
slogan.

## Doctrine (positive, not credulous)

1. **Do not reject without a structural argument.** Missing code, an untried
   baseline, or a failed first smoke is not a mathematical impossibility.
2. **Do not declare solved without evidence.** A wiring test, a finite loss, or
   a single seed is not an acceptance gate.
3. **Search for the strongest constructive route first.** Prefer hard BCs /
   SDF cages / causal marching / multilevel FBPINN / conditioned operators over
   softening the claim.
4. **Turn every caveat into a falsification test.** If you write "may fail at
   junctions", ship `test_*_junction_raises`. If you write "beats retrain", ship
   the equal-budget comparator.
5. **Iterate after a losing baseline.** Diagnosis → change → remeasure. Scope
   reduction is the *last* move, not the first.

## Exploration vs shipping (two phases)

| Phase | Allowed | Forbidden |
| --- | --- | --- |
| **A – explore** | Bold prototypes in alpha submodules, failing experiments, GPU sweeps under `$OMNIBIAS_SCRATCH` | Premature new distributions; committing huge binaries; claiming victory in docs |
| **B – ship** | Acceptance-gated APIs, parity tests, `--smoke` + multi-seed JSON, capability-matrix rows with a guarantee level | Silent scope cuts; "Honesty:" prose that excuses a missing gate |

Alpha **submodules** of `omnibias-pinn` (`train`, `domain`, `operator`, …) are
encouraged. New top-level packages still must earn independence
(`omnibias-dev-new-package`).

## PINN-specific acceptance gates

| Gap | Minimum gate |
| --- | --- |
| Causality | Equal-budget arms (whole / causal / marching / combined) on a named PDE; multi-seed reference + seam error; advance gate never silently promotes an unconverged window; same-time triviality guard |
| Geometry | Negative-inside CSG truth table; SDF-driven solver sampling; Dirichlet hard cage on smooth primitives; Neumann/Robin only where normals exist; vs soft-penalty baseline |
| Operators | Held-out params/shapes; conditioned vs unconditioned vs per-instance PINN retrain; amortization break-even reported |
| Spectral bias | Nonzero NTK spectra on a known linear model; equal-param arms (plain / Fourier / Mscale / FBPINN); mode-wise error + task alignment |

Certificates: reuse a-posteriori linear PDE certificates when stability
constants exist; for nonlinear cases seal **residual evidence** without
relabeling it a solution bound.

## Claim language

Lead with achieved capability, then state:

- **Guarantee level** — `by construction` / `empirical (multi-seed)` /
  `certified (sound enclosure)` / `unverified prototype`
- **Acceptance domain** — PDE family, geometry class, budget, seeds

Never substitute smoke JSON for a full multi-seed distribution.

## Where the code lives

- Causal marching: `omnibias.pinn.train`
- SDF / hard curved BCs: `omnibias.pinn.domain`
- DeepONet / FNO conditioning: `omnibias.pinn.operator`
- FBPINN / NTK / bands: `omnibias.pinn.{torch,jax}.fields.fbpinn`,
  `omnibias.pinn.{torch,jax}.losses.ntk`, `SpectralBandScheduler`
- Benchmarks: `benchmarks/{causal_marching,geometry_sdf,operator_zero_shot,spectral_bias_fbpinn}.py`
