---
name: omnibias-dev-pinn-research
description: Ambitious, falsifiable PINN research inside omnibias-pinn -- separate structural impossibility from absent implementation, turn every caveat into an acceptance test, iterate after a losing baseline, and claim success the moment the gate passes. Use when closing PINN capability gaps (causality, geometry, operators, spectral bias), designing benchmarks, or deciding whether a limitation is real. For contributors modifying omnibias itself, not for consumers using it.
---

# PINN research doctrine (discovery + absolute gates)

Closed-form activation derivatives and exact hard cages are *leverage*. Use them
to dissolve named limitations on named problem families. You are **authorized to
claim a capability plainly the moment its absolute gate passes**.

## Doctrine (constructive by default)

1. **Default to achievable.** Missing code, an untried baseline, or a failed
   first smoke is not a mathematical impossibility -- hunt the constructive
   route first (hard cages, closed-form towers, one-shot collocation,
   causal marching, conditioned operators).
2. **You earn the claim when the gate passes.** Capability claims need an
   absolute acceptance gate: multi-seed empirical result with skill > 0,
   by-construction identity, or a sound certificate. Relative comparisons
   between two failing arms do not count.
3. **Search for the strongest constructive route first.** Prefer hard BCs /
   SDF cages / causal marching / multilevel FBPINN / conditioned operators /
   one-shot least-squares over softening the claim.
4. **Turn every caveat into a falsification test.** If you write "may fail at
   junctions", ship `test_*_junction_raises`. If you write "beats retrain", ship
   the equal-budget comparator with an absolute skill floor.
5. **Iterate after a losing baseline.** Diagnosis → change → remeasure. Scope
   reduction is the *last* move, not the first.

## Exploration vs shipping (two phases)

| Phase | Allowed | Forbidden |
| --- | --- | --- |
| **A – explore** | Bold prototypes in alpha submodules, failing experiments, GPU sweeps under `$OMNIBIAS_SCRATCH` | Premature new distributions; committing huge binaries; claiming victory before a gate exists |
| **B – ship** | Acceptance-gated APIs, parity tests, smoke + multi-seed JSON with a `gates` block, capability-matrix rows with a guarantee level | Silent scope cuts; prose that excuses a missing gate |

Alpha **submodules** of `omnibias-pinn` (`train`, `domain`, `operator`, …) are
encouraged. New top-level packages still must earn independence
(`omnibias-dev-new-package`).

## Validity floor (protects discoveries)

An absolute gate is what lets you tell a breakthrough from a diverged
optimizer. Worked case: the parametric heat operator benchmark marched with
explicit RK4, produced `max|u| ~ 1e9` at diffusivity 0.24, still passed
`isfinite`, and every arm reported MSE ~1e17. Switching to ETDRK4 (exact
linear advance) plus a maximum-principle guard turned an invalid experiment
into a real one. Always ask, in order:

1. Is the **reference** physically valid?
2. Does every arm beat the **zero predictor** (`skill_score > 0`)?
3. Does absolute error clear a **named threshold**?

Emit a `gates` block in every artifact (`benchmarks/_gates.py`).

## PINN-specific acceptance gates

| Gap | Constructive route | Absolute gate |
| --- | --- | --- |
| Causality | Hard IC/BC cage (heat) + soft-IC Krishnapriyan reaction; closed-form residual; gated `march_solve`; IC via `ic_fn` on marcher `slice_points` (never linspace-ordered `ic_values`) | Every arm `skill_score > 0`; **heat** best-arm median rel-L2 clears a named threshold (whole-interval may win); **reaction** best marching arm beats `whole_interval`; `advance_policy="gate"` |
| Geometry | Negative-inside SDF + `DistanceConstrainedField` | Boundary identity by construction; hard interior skill > 0 and beats soft-penalty |
| Operators | Multi-head DeepONet + ETDRK4 reference | Maximum principle on slabs; every arm skill > 0; conditioned median rel-L2 beats unconditioned **and** per-instance retrain |
| Spectral bias | One-shot frozen-feature least-squares (no GD dynamics) | `lstsq` rel-L2 < 5e-6 through f=16; capacity falsification at high f by raising feature count |

Regime note: marching is not a free win. On linear heat with a hard cage,
whole-interval can already hit skill near 1; on stiff reaction
(`u_t = rho u(1-u)`, `rho=12`) whole-interval fails causality and gated
marching is the gate that must win. See `benchmarks/causal_marching.py`.

Certificates: reuse a-posteriori linear PDE certificates when stability
constants exist; for nonlinear cases seal **residual evidence** and label it
as such — then keep improving until a solution bound is earned.

## Claim language

Lead with achieved capability, then state:

- **Guarantee level** — `by construction` / `empirical (multi-seed)` /
  `certified (sound enclosure)` / `unverified prototype`
- **Acceptance domain** — PDE family, geometry class, budget, seeds

Never substitute smoke JSON for a full multi-seed distribution when claiming
a multi-seed result. When the gate passes, say so plainly.

## Where the code lives

- Causal marching: `omnibias.pinn.train`
- SDF / hard curved BCs: `omnibias.pinn.domain`
- DeepONet / FNO conditioning: `omnibias.pinn.operator`
- FBPINN / NTK / bands: `omnibias.pinn.{torch,jax}.fields.fbpinn`,
  `omnibias.pinn.{torch,jax}.losses.ntk`, `SpectralBandScheduler`
- One-shot collocation: `omnibias.pinn.solver.torch.solve_least_squares`
- Benchmarks: `benchmarks/{causal_marching,geometry_sdf,operator_zero_shot,spectral_bias_fbpinn}.py`
- Shared gates: `benchmarks/_gates.py`
