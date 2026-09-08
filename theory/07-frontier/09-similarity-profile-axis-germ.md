# 07-09 Similarity profile, axis germ, order-n sources

## 1. Thesis and status

The paper’s first object is a PDE for `(E, U, Π)` in similarity
coordinates `(X, η)` with operators `T_b`, `Z_b` from Lemma 4.1, exact
incompressibility via `V_0`, and `Π_X = E²/(2X)`. Axis regularity is a
germ of `E / sqrt(2X)` at `X = 0`. Background coefficients at order
`q^{2nh}` are sourced by jets of lower-order products. Omnibias already
has those three registers; this spec evaluates **this** residual.

- **Status**: shipped (G1–G5 CI; founding bias collapse, not temperature collapse; not Clay (A)/(B); not a forced-blowup reproof)
- **Depends on**: 07-08, 01-10
- **Blocks**: none

## 2. Where it lands

`omnibias.pinn.certified.anisotropic` in `omnibias-pinn`. No new
package: the object is a certified PINN fragment, the same domain and
audience as 07-02. Core stays numpy-free; this module is Fraction /
Interval only. Discovery still lives in
`omnibias.pinn.jax.discovery.euler3d_axisym` and now consumes the
locked residual instead of a random proxy.

## 3. Prior art in omnibias

- `omnibias.core.verified.taylor_model.TaylorModel`
- `omnibias.core.verified.jet_mv.jet_multiply`
- `omnibias.pinn.certified.navier_stokes` — compactified axisymmetric
  metadata, not this residual. Do not grow that 9.7k-line file.
- `omnibias.pinn.certified.weak_form` — CCF / weak-form box. Wrong lane.

**Confirmed gap.** Lemma 4.1 operators, the axis germ of `E/sqrt(2X)`,
and order-`n` source jets were not evaluated on a locked axis-regular
profile.

## 4. Mathematics

`L = 1 - 2h η²`, `d = 1 - η²`, `D_X f = X ∂_X f`, `D = 1/2 - h`:

```
T_b f = L^{-1} (-b f + D η f_η + D_X f)
Z_b f = L^{-1} (2b η f + d f_η - 2η D_X f)
```

Axis regularity: `E = sqrt(2X) F` with `F` smooth. The locked plant
takes `F = c(1 + a X)` so the Taylor remainder of `F` at `X = 0` is
`{0}`. Stress `T` is the radial integral of a polynomial residual `R`;
`R = -div_r T` is FTC over `Q`. Sources are the Cauchy product of two
jets, independently via `jet_multiply`.

This is founding **bias-collapse** arithmetic (exact identities). It is
not temperature collapse.

## 5. Worked example

`h = 1/200`, `c = 1`, `a = 1`, `X = 1`, `η = 0`, `b = 0`. Then
`F = 2`, `F_X = 1`, `L = 1`, `T_0 F = 1`, `Z_0 F = 0`. At `η = 1/2`,
`L = 399/400`, `T_0 F = 400/399`, `Z_0 F = -400/399`. Residual plant
`R = r²` has `T = -r³/5` and `R + div_r T = 0` at `r = 1`. Axis germ
remainder is `{0}`. Order-1 product of `1 + X` is `1 + 2X`.

## 6. Proposed API

```
SimilarityScales(h)
profile_operators(h) -> T_b, Z_b
locked_axis_regular_profile()
leading_tangential_residual(profile, X, eta)
axis_germ(profile, order)
coefficient_source_jet(lower, order)
```

Honesty payload: all NS proof flags false;
`forced_blowup_reproof_claim=False`. No torch/jax in this module.

## 7. Practical use cases

- Lock a hand-checkable profile before any numerical discovery loop.
- Replace a random residual proxy with an exact identity.
- Feed axis germs into later joining / moment matching (those stay
  external).
- Source higher-order background coefficients from Cauchy products.

## 8. Acceptance gates

- **G1.** Operators vs Lemma 4.1 on the locked profile, exact `Fraction`.
- **G2.** Residual `= -div T` exactly.
- **G3.** Axis TM remainder is `{0}`.
- **G4.** `jet_multiply` source matches the Cauchy product.
- **G5.** Honesty / no parent flag.

## 9. Benchmark plan

`benchmarks/anisotropic_profile.py` writes
`docs/benchmarks/anisotropic_profile_smoke.json`. Algebra and honesty
run in CI.

## 10. Honesty and scope

Not Theorem 4.6 (heat exterior + cone + moment matching). Those are
07-10 / 07-12. Not a second proof of Clay (C)/(D). Unforced (A)/(B)
stays external. `navier_stokes_proof_claim` is never set by hand.
Forbidden: “we prove global regularity”, “we reproduce the OpenAI
proof”.

## 11. Open questions and risks

- Profile-joining moments and `N log X` shear stay external.
- A float FFT residual is not this gate.
- Falsifier: a locked `Fraction` identity disagrees with Lemma 4.1.

## 12. Implementation checklist

- [x] `omnibias.pinn.certified.anisotropic`
- [x] `packages/omnibias-pinn/tests/certified/test_anisotropic_profile.py`
- [x] `euler3d_axisym` residual swap
- [x] `benchmarks/anisotropic_profile.py` plus smoke JSON
- [x] Docs page and nav entry
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: Navier-Stokes forced blowup (Clay C/D)**, already true in the
world. This fragment re-derives Lemma 4.1 operators, an axis germ, and
a source jet. It does not claim a second proof. Unforced Navier-Stokes
regularity (Clay A/B) stays an external obligation. The parent-level
honesty flag stays false.
