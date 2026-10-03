# Hilbert XVI: actual slow-line zeta and a Cauchy majorant

This companion extracts `zeta` from the quadratic embedding at `r = -1`,
on the slow line `h = 0`. The identities are exact. A rectangular
`ComplexInterval` enclosure supplies a Cauchy majorant for the divided
remainder `Z` on the slice `lambda0 = lambda1 = 0`. That slice is not
the fold compact of `(L, lambda1)`, so the bound does **not** pass G1.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. Algebraic embedding

On `h = 0`,

    V = 1 - v - nu v^2,    nu v0^2 + v0 - 1 = 0.

Once `v0` is a root,

    V = (v0 - v) (1 + nu (v0 + v)).

For `lambda0 = lambda1 = 0` the slow-line field is cubic and

    zeta = - ell l (v + 2 v0) / (3 v0 D^2),

with `ell = 1 + 2 nu v`, `l = 1 + 2 nu v0`, `D = 1 + nu (v0 + v)`,
and `k = 3 v0 / l`, `eps = nu k`. Substituting `v = v0` gives
`zeta = -1`. At `nu = 0` one has `v = 1 - V` and
`zeta(V, 0) = -1 + V/3`. The linear jet at `V = 0` is
`1/(3 v0 l) = 1/(k l^2)`.

The same embedding with the implicit `k` equation

    k = 3 v0 / l + nu^2 k^2 lambda1 / l^2 + 4 nu^4 k^3 lambda0 / l^4

is exact at a rational sample with `lambda1 = 288/125`, `k = 2`.
The cubic field recovers the closed-form `zeta` at `lambda = 0`, and
`Vdot` at `V = 0` is `eps^3 lambda0` for any lambda. A Cauchy
majorant for `Z` is sealed only on the `lambda = 0` slice.

## 2. Division and Cauchy majorant

Joint vanishing on the axes `V = 0` and `nu = 0` yields a holomorphic

    Z = (zeta + 1 - V/3) / (eps V)

at `(0, 0)`. Enclosing `zeta + 1 - V/3` on a rectangle containing the
polydisc `|V| <= 1/8`, `|nu| <= 1/16` produces a finite majorant `M`.
Schwarz on those axes then bounds `|Z| <= M / (R E k_min)` on the
slice. A rational sample lies below the bound.

The fold remainder of
[HILBERT16-FOLD-LEADING.md](HILBERT16-FOLD-LEADING.md) uses a compact of
`(L, lambda1)`, sealed separately in
[HILBERT16-FOLD-ZETA.md](HILBERT16-FOLD-ZETA.md). Physical C2 of `log D'`,
outgoing first-hit, complete first-hit, and Stage B/C continuation stay
open.

Lean: [Hilbert16CanonicalZeta.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16CanonicalZeta.lean).
Python: `omnibias.dynamics.canonical_zeta`.

## 3. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_canonical_zeta.py -q
uv run --no-sync python -m benchmarks.hilbert16_canonical_zeta
lake build OmnibiasAnalytic.Dynamics.Hilbert16CanonicalZeta
```
