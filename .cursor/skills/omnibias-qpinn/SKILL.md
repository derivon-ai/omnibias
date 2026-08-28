---
name: omnibias-qpinn
description: Build quantum PINNs with closed-form residuals for Schrödinger, Gross-Pitaevskii, Helmholtz, Klein-Gordon, and Dirac, plus norm and Hermitian cages. Use when inventing a new quantum residual or a vortex diagnostic on the field substrate.
---

# omnibias-qpinn

Quantum-physics-informed neural networks with closed-form n-th derivative operators on
top of omnibias-pinn. Cross-backend (torch + jax) residuals for Schrodinger /
Gross-Pitaevskii / rotating-frame Gross-Pitaevskii / Helmholtz / Klein-Gordon / Dirac,
plus norm-conservation, Bloch-periodic, and Hermitian-operator cages, parity-projection
helpers, and vortex / Thomas-Fermi diagnostics for 2D rotating condensates.

## Why nested AD fails

Complex nested AD through a Laplacian-plus-potential residual is the same order
bottleneck as classical PINNs, with extra phase wrapping. Generic QPINNs have no
closed-form tower and no Hermitian cage.

## What only this tower unlocks

Quantum-physics-informed neural networks with closed-form n-th derivative operators on
top of omnibias-pinn. Cross-backend (torch + jax) residuals for Schrodinger /
Gross-Pitaevskii / rotating-frame Gross-Pitaevskii / Helmholtz / Klein-Gordon / Dirac,
plus norm-conservation, Bloch-periodic, and Hermitian-operator cages, parity-projection
helpers, and vortex / Thomas-Fermi diagnostics for 2D rotating condensates.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Builds on `omnibias-pinn` / `omnibias-fields`.

| You want | Import |
| --- | --- |
| TISE / TDSE / GPE residuals | `omnibias.qpinn` |
| Example | `docs/examples/qpinn_tise_qho.py` |

## Extend

- Source: [`packages/omnibias-qpinn`](../../../packages/omnibias-qpinn).
- Namespace: `omnibias.qpinn`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-qpinn/tests -q`.
- Compose with `omnibias-pinn`, `omnibias-fields`, `omnibias-ferminet` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A rotating GPE whose closed-form residual, norm cage, and vortex diagnostic pass an
absolute skill gate against a spectral baseline.


## Bakeoffs

Nested-AD cost and closed-form accuracy live in
[`docs/benchmarks/`](../../../docs/benchmarks/):
`laplacian_scaling.json`, `polylaplacian_order.json`,
`derivative_order.json`, `jet_vs_nested_ad_smoke.json`.
Heavy regeneration follows the workspace compute rule, not a skill taboo.

## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
