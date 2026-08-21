---
name: omnibias-frontier
description: Use omnibias for frontier sub-results -- certified fluids, CAP singularities, gauge/spectral enclosures, SOS positivity, validated dynamics -- while keeping Clay / Nobel parent problems as external obligations unless a finite gate passes. Use when building on omnibias toward ambitious scientific claims, or when asking whether a local certificate implies a famous conjecture.
---

# Frontier sub-results with omnibias

omnibias can discharge **finite or compact** obligations that sit under famous
open problems. It does not prove the famous parents by proximity.

## What you can claim

- Local / short-time fluid enclosures and certified residual evidence
  (`omnibias.pinn` certified NS; CCF CAP cookbooks)
- Eigenvalue lower bounds and SOS positivity (`omnibias.core.verified.eig`,
  `omnibias-sos`)
- Validated ODE / PDE flow (`omnibias-dynamics`, Lohner / TM)
- Gauge / geometry primitives with sealed local scope
  (`omnibias.geometry.gauge`)
- Dirichlet / zeta enclosures on `Re(s) > 1` only
- Finite Keller-map identities and a blind deg-2 / deg-3 tangent sweep
  (`omnibias.holonomic.keller` via `run_discovery`) — a map in the sweep
  family, not a Jacobian-conjecture proof
- Jacobian `n=2` finite box `C_box(d,h,G)` (`omnibias.holonomic.jacobian_n2`)
  — identical constant Jacobian plus a rational grid collision or a
  Gabber inverse-degree failure. A miss is not the parent.
  `jacobian_n2_claim` only via `escalate_n2_result`
- DGG / Rybin H* cost separation (`omnibias.combinatorics.unsplittable`) —
  a separator in the H* family; the congestion theorem stays true. A
  capped DAG ≤6 miss is `BLOCKED`, not a parent proof
- Triangle-free colourings and template-graph predicates
  (`omnibias.combinatorics.ramsey` / `.extremal`) — not Erdős 183 / 146 / 180
- The discovery engine (`omnibias.core.proof.discovery` + `.catalog` +
  `.condition` + `.observe`): a named family, a typed
  `ConditionHypothesis` in a finite grammar, or a shared
  `Observation` through `select_class` / `discover_observation`.
  Pack tables with `omnibias.symbolic.ingest` (tags optional). Soft
  residuals are not certificates. Uniqueness is span-scoped. An
  incomplete grammar miss is `search_incomplete`, never “no condition
  exists.”

Each claim needs an absolute gate: multi-seed skill > 0, by-construction
identity, or a sound enclosure. Prefer a `gates` block in any public artifact.

## What you must not claim

- Riemann Hypothesis proved or inferred from `Re(s) > 1` enclosures
- Navier-Stokes global regularity
- Yang-Mills mass gap solved
- P = NP (or P ≠ NP) from discrete / submodular packages
- That Lean discharged a continuum or asymptotic obligation
- `theorem_prover_verified` / `mathlib_verified` without a genuine `lake build`
- That omnibias refuted Keller, Goemans, or Erdős 183 / 146 / 180
- `ten_proofs_formalization_claim` (we did not `lake build` their Lean)
- Jacobian `n=2` settled

## Escalation

prototype → empirical multi-seed → sound enclosure → kernel verified →
Mathlib verified. Label the guarantee level and acceptance domain next to
every headline number.

## Start here

- Cookbooks: `docs/cookbook/navier-stokes-certified.md`,
  `ccf-singularity.md`, `euler2d-vortex.md`, `sqg-vortex.md`,
  `proof-carrying-fluid-dynamics.md`, `keller-jacobian.md`,
  `unsplittable-flow.md`, `finite-ramsey-colouring.md`,
  `extremal-graph-replay.md`, `discovery-loop.md`,
  `docs/api/discovery.md`, `proof-machine.md`
- Capability matrix: `docs/benchmarks/pinn_four_gap_matrix.md`
- Formal loop: `docs/scope-and-guarantees.md`, `omnibias.core.proof`
