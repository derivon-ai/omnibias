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
`HILBERT16-CHART-CELLS.md`. The canonical slow-line zeta / Cauchy
majorant is `HILBERT16-CANONICAL-ZETA.md`. The fold-compact Cauchy
majorant is `HILBERT16-FOLD-ZETA.md`. Frozen-Z C2 identities are
`HILBERT16-PHYSICAL-C2.md`. The unfrozen-Z `Z_x` first-log-derivative
gap is `HILBERT16-Z-X-GAP.md`. The holomorphic `Z_v` bound is
`HILBERT16-Z-V-BOUND.md`. The slow-line `Z_V` chain is
`HILBERT16-Z-SLOW-V.md`. The matching-chart fold I-map `Z_x` bound is
`HILBERT16-FOLD-Z-X.md`. The kill-line Stage-B height inflation is
`HILBERT16-STAGE-B.md`. The kill-line Stage-A shrinking-rectangle
wall is `HILBERT16-STAGE-A.md`. The kill-line `chi_b` threshold is
`HILBERT16-CHI-B.md`. The kill-line `dx_e` leading factors are
`HILBERT16-DX-E-LEADING.md`. The kill-line uniform-in-`chi` `dx_e`
majorant is `HILBERT16-DX-E-UNIF.md`. The kill-line Stage-C `a_min`
floor is `HILBERT16-STAGE-C.md`. The kill-line Stage-C exit energy is
`HILBERT16-STAGE-C-EXIT.md`. The kill-line Stage-C leading `T_h` floor
is `HILBERT16-STAGE-C-TH.md`. The kill-line Stage-C start gap at `y_1=1`
is `HILBERT16-STAGE-C-GAP.md`. The kill-line Stage-C C=0 `T(h)` envelope
is `HILBERT16-STAGE-C-ENV.md`. The kill-line Stage-C C=2 integrating
factor is `HILBERT16-STAGE-C-IF.md`. The kill-line Stage-C C=2 `T(h)`
majorant is `HILBERT16-STAGE-C-INT.md`. The kill-line Stage-C C=2 lower
`T(h)` envelope is `HILBERT16-STAGE-C-LO.md`. The kill-line Stage-C C=2
tight `T(h)` ratio is `HILBERT16-STAGE-C-K.md`. The kill-line Stage-C C=2
`T-h` bootstrap is `HILBERT16-STAGE-C-BOOT.md`. The kill-line Stage-C
continuation rectangle is `HILBERT16-STAGE-C-RECT.md`. The kill-line
Stage-C comparison first-hit of `h=1` is `HILBERT16-STAGE-C-HIT.md`.
The kill-line Stage-C comparison first-hit of `E_out` is
`HILBERT16-STAGE-C-SEC.md`. The kill-line Stage-C Lohner first-hit of
matching-chart `x=4` from Stage-C start is
`HILBERT16-STAGE-C-ONESHOT.md`. The shrinking-eps Stage-C Lohner pack
is `HILBERT16-STAGE-C-ONESHOT-EPS.md`. The parametric-eps Stage-C
Lohner cover of `[23/400, 1/16]` is `HILBERT16-STAGE-C-EPS-SPAN.md`.
The chart-O matching-chart Lohner pack is
`HILBERT16-STAGE-C-ORIGIN.md`. The parametric-sep chart-O Lohner
cover of `[3/2, 2]` is `HILBERT16-STAGE-C-ORIGIN-SPAN.md`. The
nearer-interface cover of `[7/4, 2]` from `x=1/8` is
`HILBERT16-STAGE-C-ORIGIN-IFACE.md`. The `x=1/16` cover of
`[15/8, 2]` is `HILBERT16-STAGE-C-ORIGIN-NEAR.md`. The `x=1/32`
cover of `[31/16, 2]` is `HILBERT16-STAGE-C-ORIGIN-X32.md`. The
uniform-in-`r1` comparison first-hit on `eps` in `[1/32, 1/16]` is
`HILBERT16-STAGE-C-COMPARE.md`. The comparison first-hit for every
`eps` in `(0, 1/16]` is `HILBERT16-STAGE-C-UNIFORM.md`. The comparison
first-hit from every start in `(0, 1/2]` is `HILBERT16-STAGE-C-INTERFACE.md`.
The kill-line `sep * S_pre` bound on every `sep` in `(0, 1]` is
`HILBERT16-SEP-SPRE.md`. The `dx_e` factors on `lambda1` in
`[-4, -2]` are `HILBERT16-DX-E-OFF.md`. The same factors for every
`lambda1 <= -2` are `HILBERT16-DX-E-RAY.md`. The same factors for
`lambda1` in `[-3/2, -2)` are `HILBERT16-DX-E-NEAR.md`. The same
factors for every `lambda1` in `(-3/2, 0)` are `HILBERT16-DX-E-OPEN.md`.
The intrinsic eta-section obstruction at D-C and chart O is
`HILBERT16-WEIGHTED-SECTION.md`.
The frozen-section one-scale no-go and its moving-section counterexample are
`HILBERT16-QUASIHOMOGENEOUS-DICHOTOMY.md`.
The direct `tau`/log-W bounded-format obstruction and open normalized route are
`HILBERT16-LN-FORMAT-BARRIER.md`.
The complex fixed-real-time flow, local event branch, regular-event cover, and
eight-cell physical outgoing `E_out` cover across the cubic model's `sep=0` /
`r1=0` limits, which still lack the incoming branch, full physical singular
return family, and LN membership, are
`HILBERT16-COMPLEX-NORMAL-FLOW.md`.
The conditional Picard--Fuchs/Abelian zero-count transfer and open DRR
physical-return premises are `HILBERT16-ABELIAN-DRR-TRANSFER.md`.
The finite quadratic Bautin-jet stabilization and exact all-orders inference
barrier are `HILBERT16-BAUTIN-STABILIZATION-BARRIER.md`.
The Songling four-cycle reproduction-readiness and binary64 precision barrier
are `HILBERT16-SONGLING-LOWER-BOUND.md`.
The Part-A exact polygonal barrier and reduced Positivstellensatz audit is
`HILBERT16-PART-A-POLYGON-SOS.md`.
The shrinking-root x-corridor is
`HILBERT16-OUTGOING-CORRIDOR.md`. The post-corridor `(V,h)` hypotheses
are `HILBERT16-POST-CORRIDOR.md`. The alpha-0 `T-h` envelope is
`HILBERT16-HEIGHT-ENVELOPE.md`. The `C=2` leading `|q|` ratio is
`HILBERT16-Q-RATIO-C2.md`. The `C=0` `k=1+O(nu)` jet is
`HILBERT16-K-ZETA-REMAINDER.md`. The kill-compact `Z` majorant is
`HILBERT16-KILL-ZETA.md`. The cancelled-N holomorphic `Z` bound is
`HILBERT16-CANCELLED-N.md`. The `C!=0` height-mix identities are
`HILBERT16-HEIGHT-MIX.md`. The `T_h`-gap identities are
`HILBERT16-ORBIT-TH.md`. The comparison-bootstrap `T-h` integral is
`HILBERT16-TH-INTEGRAL.md`. The cubic `(V,h)` Lohner orbit is
`HILBERT16-VH-ORBIT.md`. Matching-chart `E_out` first-hit is
`HILBERT16-E-OUT-SECTION.md`. The shrinking-eps `E_out` pack is
`HILBERT16-E-OUT-EPS.md`. The kill-line comparison speed bound is
`HILBERT16-E-OUT-SPEED.md`. The incoming GRAZING comparison speed bound is
`HILBERT16-E-SIGMA-SPEED.md`. The incoming `V=1/4` first-hit is
`HILBERT16-E-SIGMA-IN.md`. The declared-point `E_sigma` first-hit is
`HILBERT16-E-SIGMA-HIT.md`. The comparison GRAZING `E_sigma` zero from
`V=0` is `HILBERT16-E-SIGMA-FROM0.md`. The uniform cancelled-height
comparison is `HILBERT16-E-SIGMA-UNIF.md`. The orbit-aligned wall
`E_sigma` hit is `HILBERT16-E-SIGMA-WALL.md`. The wall-box `h`-interval
cover is `HILBERT16-E-SIGMA-BOX.md`. The L=0 whole-wall `h`-span
cover is `HILBERT16-E-SIGMA-SPAN.md`. The L-pack wall-span cover is
`HILBERT16-E-SIGMA-PACK.md`. The shrinking-eps aligned pack is
`HILBERT16-E-SIGMA-EPS.md`. The one-shot Lohner-from-`V=0` hit is
`HILBERT16-E-SIGMA-ONESHOT.md`. The shrinking-eps one-shot pack is
`HILBERT16-E-SIGMA-ONESHOT-EPS.md`. The compact aligned parametric-eps
cover is `HILBERT16-E-SIGMA-EPS-SPAN.md`. The lower aligned parametric-eps
cover is `HILBERT16-E-SIGMA-EPS-LO.md`. G1 and G4 remain failed there, and
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
