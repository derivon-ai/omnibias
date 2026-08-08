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

Each claim needs an absolute gate: multi-seed skill > 0, by-construction
identity, or a sound enclosure. Prefer a `gates` block in any public artifact.

## What you must not claim

- Riemann Hypothesis proved or inferred from `Re(s) > 1` enclosures
- Navier-Stokes global regularity
- Yang-Mills mass gap solved
- P = NP (or P ≠ NP) from discrete / submodular packages
- That Lean discharged a continuum or asymptotic obligation
- `theorem_prover_verified` / `mathlib_verified` without a genuine `lake build`

## Escalation

prototype → empirical multi-seed → sound enclosure → kernel verified →
Mathlib verified. Label the guarantee level and acceptance domain next to
every headline number.

## Start here

- Cookbooks: `docs/cookbook/navier-stokes-certified.md`,
  `ccf-singularity.md`, `euler2d-vortex.md`, `sqg-vortex.md`,
  `proof-carrying-fluid-dynamics.md`
- Capability matrix: `docs/benchmarks/pinn_four_gap_matrix.md`
- Formal loop: `docs/scope-and-guarantees.md`, `omnibias.core.proof`
