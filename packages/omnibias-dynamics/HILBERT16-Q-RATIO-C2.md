# Hilbert XVI: uniform C=2 leading |q| ratio on chart O

This companion continues [the alpha-0 envelope](HILBERT16-HEIGHT-ENVELOPE.md).
The written first-root comparison uses `|q| <= C epsilon (epsilon^2 + T)`.
That bound was stated at the matching section. On the shrinking-root
sequence `lambda1 = -2` it is a **V-only** algebraic fact: the leading
ratio

    |q| / (epsilon (epsilon^2 + T)) = |x^2 + lambda1 x + L| / (1 + x^2 / 2)

does not depend on `epsilon` or on `h`. Between the slow-line roots it
equals `(x-r1)(r2-x)/(1+x^2/2)`. The gap

    2 + x^2 - (x-r1)(r2-x) = 2(x - 1/2)^2 + 3/2 + L

is at least `3/2 + L > 0`. The discriminant of that quadratic is
`lambda1^2 - 12 L - 16 = -12(1+L) < 0`. Past the second root the outer
gap is `2 + 2x - L`, positive for `L < 2`. Hence `C = 2` is uniform in
`x` and in `r1 -> 0`. As `x -> infinity` (the large-section scaling
`V = O(1)`, `x = V/epsilon`) the ratio tends to `2` from below.

This bounds leading `|q|` as a function of `V` only. It is not
`k = 1 + O(epsilon)`, a `zeta` remainder, the transversal event, G1, or
Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. Why the bound does not need an h-bootstrap

At leading order `q` is the cubic `-f(V)` and does not depend on `h`.
The comparison coefficient `C` is therefore decided on the `V`-line.
The matching-section ratio of [the envelope note](HILBERT16-HEIGHT-ENVELOPE.md)
is the restriction of this same function to `x = x_*`.

## 2. What remains for G1

- `k = 1 + O(epsilon)` and the `zeta` remainder versus the cubic, on a
  compact `(V,h)` rectangle: the `C=0` jet and cubic prefactor are
  [HILBERT16-K-ZETA-REMAINDER.md](HILBERT16-K-ZETA-REMAINDER.md); a
  rectangular `Z` majorant including `L=0` is
  [HILBERT16-KILL-ZETA.md](HILBERT16-KILL-ZETA.md); a cancelled-N
  holomorphic bound on the slow line is
  [HILBERT16-CANCELLED-N.md](HILBERT16-CANCELLED-N.md); `C != 0` `|g_h|`
  is [HILBERT16-HEIGHT-MIX.md](HILBERT16-HEIGHT-MIX.md).
- Height-section first-hit of the selected large first-root section
  (the transversal event).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16QRatioC2.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16QRatioC2.lean).
Python: `omnibias.dynamics.q_ratio_c2`.

## 3. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_q_ratio_c2.py -q
uv run --no-sync python -m benchmarks.hilbert16_q_ratio_c2
lake build OmnibiasAnalytic.Dynamics.Hilbert16QRatioC2
```
