# Hilbert XVI: k=1+O(ε) and the cubic remainder prefactor

This companion continues [the C=2 |q| ratio](HILBERT16-Q-RATIO-C2.md).
The written first-root comparison uses `T_h = q/h + k` with
`k = 1 + O(epsilon)` and a cubic `zeta` remainder versus
`-1 + V/3`. On the `C = 0` normal chart those are algebraic:

    k_normal = (1 + 2 nu v)(1 - A nu v).

At `A = 1` this is `1 + nu v - 2 nu^2 v^2`. The first-order jet
`k_lead = 1 - nu (V-1)` matches it through order `nu` on
`V = 1 - v - nu v^2`, with exact remainder `-3 nu^2 v^2`. On
`|v| <= 2` and `nu <= 1/8`, `|k_normal - 1| <= 3 nu`.

The cubic correction past the two-root leading term is

    q_cubic = epsilon^3 (x-r1)(x-r2) + epsilon^4 x^3 / 3

at `V = -epsilon x`. Relative to `epsilon (epsilon^2 + T)` with
`T = V^2/2`, the prefactor identity

    2 (epsilon^2 + T) - w^2 = 2 epsilon^2

bounds that correction by `2 epsilon |V|` times a holomorphic `Z`
factor. This is not a bound on `Z`, not `T-h = O(epsilon)` along the
actual orbit, not the transversal event, G1, or Hilbert XVI. Height
mixing `|g_h| = O(epsilon^2)` at `C != 0` is not this slice.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. Why k does not need an h-bootstrap on C=0

At `C = 0` the normal coefficient `k` is independent of `h`. The
`O(nu)` bound is a polynomial in `(nu, v)` on a compact `v`-interval.

## 2. What remains for G1

- A `Z` bound on the kill sequence (the relative prefactor is not `|Z|`;
  a rectangular majorant including `L=0` is
  [HILBERT16-KILL-ZETA.md](HILBERT16-KILL-ZETA.md); a cancelled-N
  holomorphic bound with `2 eps |V| |Z| < 1` on the slow line is
  [HILBERT16-CANCELLED-N.md](HILBERT16-CANCELLED-N.md), not `T-h`
  along the orbit or `C != 0` `|g_h|`).
- Height mixing `|g_h| = O(epsilon^2)` at `C != 0` is
  [HILBERT16-HEIGHT-MIX.md](HILBERT16-HEIGHT-MIX.md).
- `T - h = O(epsilon)` along the actual `(V,h)` orbit.
- Height-section first-hit of the selected large first-root section
  (the transversal event).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16KZetaRemainder.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16KZetaRemainder.lean).
Python: `omnibias.dynamics.k_zeta_remainder`.

## 3. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_k_zeta_remainder.py -q
uv run --no-sync python -m benchmarks.hilbert16_k_zeta_remainder
lake build OmnibiasAnalytic.Dynamics.Hilbert16KZetaRemainder
```
