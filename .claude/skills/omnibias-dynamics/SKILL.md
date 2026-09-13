---
name: omnibias-dynamics
description: Run validated variational / monodromy flow, Poincare-section enclosures, certified Lyapunov bounds, and radii-polynomial periodic-orbit proofs. Use when inventing a computer-assisted ODE/PDE existence argument on the closed-form variational tower.
---

# omnibias-dynamics

Computer-assisted dynamics: validated variational / monodromy flows, Poincare-section
enclosures, certified Lyapunov-exponent bounds, and rigorous periodic-orbit existence
via the radii-polynomial approach. Built on the omnibias closed-form variational tower
and the QR-Lohner / Newton-Kantorovich machinery in omnibias.core.verified.

## Why nested AD fails

Forward Euler plus nested AD cannot enclose a flow. Without QR-Lohner and an exact
Jacobian, wrapping inflates until the enclosure is vacuous. Generic ODE solvers return a
trajectory, not a ball that contains the true orbit.

## What only this tower unlocks

Computer-assisted dynamics: validated variational / monodromy flows, Poincare-section
enclosures, certified Lyapunov-exponent bounds, and rigorous periodic-orbit existence
via the radii-polynomial approach. Built on the omnibias closed-form variational tower
and the QR-Lohner / Newton-Kantorovich machinery in omnibias.core.verified.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Pure Python on `omnibias.core.verified` (QR-Lohner / TM).

| You want | Import |
| --- | --- |
| Variational / monodromy flow | `omnibias.dynamics` (`variational_flow`) |
| Periodic-orbit proof | `omnibias.dynamics` (`prove_periodic_orbit`) |
| Lyapunov exponent bound | `omnibias.dynamics` (`certified_lyapunov_exponent`) |
| Cone-field hyperbolicity | `omnibias.dynamics` (`certified_cone_hyperbolicity`; finite orbit; not Anosov) |
| Jet world model | `omnibias.dynamics._core.jet_world` |
| Exact-source regular stopped events and state/parameter derivatives | `omnibias.dynamics.return_maps` (`certify_stopped_event`, `verify_stopped_event`) |
| Polynomial/exponential zero bounds and actual regular return counts | `omnibias.dynamics.cyclicity` |
| Replay a finite polynomial parameter/height cover | `omnibias.dynamics.hilbert16` |

The legacy `poincare_map` reports an endpoint-sign crossing enclosure only.
Stronger regular-event claims require `return_maps`: a polynomial field and
sections generate the actual variational equations, transversality and first
eligible-event checks. Grazing or ambiguous admission remains unresolved.
`cyclicity` does not turn a fitted displacement into an exact physical one,
and `hilbert16` does not prove a graphic atlas is complete. Read
`packages/omnibias-dynamics/HILBERT16-PROGRAM.md` for the implemented evidence,
source versions, and outstanding singular-passage and coverage obligations.
The coalescing χ-atlas is `HILBERT16-COALESCING-CAPTURE.md`. The
saddle-node, shrinking-root, and two-blow-up follow-ups are
`HILBERT16-SADDLE-NODE.md`, `HILBERT16-SHRINKING-ROOT.md`, and
`HILBERT16-TWO-BLOWUP.md`. The next-atlas / scale-dichotomy findings
are `HILBERT16-NEXT-ATLAS.md`. The chart-cell ledger is
`HILBERT16-CHART-CELLS.md`. G1 and G4 remain failed there, and
`full_hilbert16_solved` stays false.

## Extend

`omnibias.dynamics.continuation.certify_segment` encloses a root uniformly for
every parameter in an interval; `certify_join` proves common-endpoint
uniqueness rather than accepting overlapping boxes. `certify_event` combines
an augmented-system root with supplied sound fold/Hopf normal-form conditions.
It does not automatically derive interval Lyapunov coefficients. Numerical
predictors and generic derivative callbacks live upstream in
`omnibias.geometry.continuation`; the certified module remains pure Python.
The analytic Bratu family and explicit continuum limits of the claim are
documented in `docs/api/neuromanifold-science.md`.

- Source: [`packages/omnibias-dynamics`](../../../packages/omnibias-dynamics).
- Namespace: `omnibias.dynamics`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-dynamics/tests -q`.
- Compose with `omnibias-verify`, `omnibias-verified-primitive`, `omnibias-control`, `omnibias-formal` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A named planar map whose unique periodic orbit is sealed by a radii polynomial and whose
monodromy spectral-radius bound feeds certified_horizon.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
