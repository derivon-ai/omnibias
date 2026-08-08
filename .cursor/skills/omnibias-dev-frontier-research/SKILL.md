---
name: omnibias-dev-frontier-research
description: Frontier research doctrine for omnibias -- decompose famous open problems into winnable finite or compact sub-obligations, escalate through absolute gates (empirical skill, sound enclosure, Lean kernel, Mathlib), and never weaken the honesty stack (RH external, no NS global regularity, no Yang-Mills mass gap, no P=NP, finite rational Lean only). Use when pursuing Nobel / Clay-adjacent sub-results, designing certified fluid / gauge / spectral / positivity claims, or deciding whether a celebrated obstruction is structural. For contributors modifying omnibias itself, not for consumers using it.
---

# Frontier research doctrine

omnibias already has moonshot *infrastructure*: CCF self-similar CAP, 2-D Euler
and SQG vortices, certified fluid dynamics, gauge transfer gap,
Lehmann-Maehly-Goerisch eigenvalue lower bounds, SOS positivity, Lohner
validated flow, `dirichlet` on `Re(s) > 1`. This skill is the doctrine that
ties them together. Ambition is encouraged; forging a claim is not.

## Decomposition rule

A famous open problem is in scope only after it is rewritten as a **finite or
compact obligation** the repo can discharge:

1. Name the celebrated statement.
2. Ask: can it be reduced to a finite rational check, a compact enclosure, a
   multi-seed empirical gate with skill > 0, or a by-construction identity?
3. If **no** -- record it as an **external obligation** in the sealed payload
   and stop. Do not infer it from a local result.
4. If **yes** -- that sub-obligation is the research object. Claim *it*, not
   the famous parent.

## Escalation ladder

| Tier | Evidence | Claim language |
| --- | --- | --- |
| 0 Unverified prototype | Code runs; no gate | "prototype; ungated" |
| 1 Empirical | Multi-seed, skill > 0, named absolute threshold, `gates` block | "empirical (multi-seed)" |
| 2 Sound enclosure | Interval / TM / SOS / Lohner contains grid **and** random sample | "certified (sound enclosure)" |
| 3 Kernel verified | Genuine `lake build` of `omnibias-verified-kernel` | `theorem_prover_verified` |
| 4 Mathlib verified | Genuine `lake build` of `omnibias-analytic` | `mathlib_verified` (never conflate with 3) |

Climb one rung at a time. Tier 1 already authorizes a plain capability claim.
Smoke JSON is not a multi-seed claim.

## Route list (frontier -> primitive)

| Frontier | Constructive sub-result route |
| --- | --- |
| Navier-Stokes regularity | Local / short-time enclosures; certified residual evidence; CCF CAP on self-similar profiles -- **not** a global-regularity claim |
| Clay Yang-Mills mass gap | Transfer-gap / spectral-gap enclosures on finite truncations; seal scope as local |
| Riemann Hypothesis | `dirichlet` enclosures on `Re(s) > 1` only; continuation and RH stay external |
| P vs NP | Exact P-class submodular minimization (Lovász / Fujishige-Wolfe); NP-hard packages keep honest non-tight gaps -- never a P=NP claim |
| Blow-up / singularity | CAP / radii-polynomial existence on a self-similar ansatz |
| Gauge theory | `omnibias.geometry.gauge` transfer / curvature primitives with sealed scope |
| Spectral geometry | Lehmann-Maehly-Goerisch lower bounds; SOS positivity certificates |

## Forbidden-claims register

Restate these disclaimers from their canonical sources; never weaken them:

| Claim that must never appear | Canonical source |
| --- | --- |
| Riemann Hypothesis proved / inferred | `omnibias.core.verified.dirichlet` docs; AGENTS.md |
| Navier-Stokes global regularity | `omnibias-pinn` certified NS docs; cookbooks |
| Yang-Mills mass gap solved | `omnibias.geometry.gauge` honesty labels |
| P = NP (or P ≠ NP) proved | `omnibias-submodular` / `omnibias-nphard` READMEs |
| Lean discharged an infinite / continuum obligation | `formal/omnibias-verified-kernel`; certificates discharge **finite rational** obligations only |
| `theorem_prover_verified` without a kernel pass | `omnibias.core.proof` -- flag is earned, never forged |
| `mathlib_verified` conflated with `theorem_prover_verified` | AGENTS.md formal-loop section |

## Worked example: absolute metrics protect discoveries

The causal-marching seam column once compared a linspace-ordered `ic_values`
vector to the marcher's random `slice_points`. Predicted artifact
`E[(sin(pi U)-sin(pi V))^2] ≈ 0.189`; measured whole-interval seams were
0.17-0.24; marching's "better" 0.046 was the same artifact averaged over four
windows. Training was fine (`ic_mode="hard"` skips the IC penalty), but the
metric was not. Fix: pass `ic_fn` evaluated on the marcher's own points.
Hard-cage seams collapsed to machine zero. **An absolute, self-consistent
metric is what separates a discovery from a diverged diagnostic.**

The same lesson as the ETDRK4 maximum-principle rescue: `isfinite` is not a
validity floor. Ask, in order -- is the reference physically valid? Does every
arm beat the zero predictor? Does absolute error clear a named threshold?

## Process

1. Write the sub-obligation and its absolute gate before coding the claim.
2. Explore under `$OMNIBIAS_SCRATCH`; commit only regenerable summaries with a
   `gates` block (`benchmarks/_gates.py`).
3. Iterate after a losing baseline. Scope reduction is the last move.
4. When the gate passes, claim the sub-result plainly with its guarantee level
   and acceptance domain. Leave the famous parent as external if it still is.
