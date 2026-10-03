# Hilbert XVI: Cauchy majorant for Z on the λ₁=-2 kill compact

This companion continues [the cubic prefactor](HILBERT16-K-ZETA-REMAINDER.md).
The relative remainder versus `C = 2` is at most `2 epsilon |V| |Z|`.
That prefactor is not `|Z|`. The shrinking-root sequence is

    lambda1 = -2,    L = r1 (2 - r1).

As `r1 -> 0`, `L -> 0`. Every two-root pair with sum 2 has product at
most 1, so the compact is `L in [0, 1]`. It **includes** the kill
limit `L = 0`.

On that compact the implicit `k` equation is a Picard contraction
around `k0 = 3 v0 / l` for small `nu`. Enclosing

    N = Vdot/eps - eps^2 L - eps lambda1 V + V^2 - V^3 / 3

on `|V| <= 0.08`, `|nu| <= 0.02` supplies a finite rectangular Cauchy
majorant `|Z| <= |N| / (R^3 E k_min)`. A real sample at `nu = 1/64`,
`r1 = 1/5`, `v = 1/2` lies below the bound. A strictly larger `nu`
box (`|nu| <= 0.05`) refuses Picard inclusion; the compact is
declared, not arbitrary.

The majorant is rectangular and not small enough for a `C = 2 + delta`
comparison. A cancelled-N holomorphic bound on the same compact is
[HILBERT16-CANCELLED-N.md](HILBERT16-CANCELLED-N.md). It is not `T-h`
along the orbit, not height-section first-hit, G1, or Hilbert XVI. The
fold compact `r in [1.4, 1.6]` is
[HILBERT16-FOLD-ZETA.md](HILBERT16-FOLD-ZETA.md).

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. Why L=0 is in the compact

On `lambda1 = -2` the roots sum to 2. Their product `L` ranges over
`[0, 1]`. The endpoint `L = 0` is `r1 = 0`, `r2 = 2`: the chart-O
kill limit. Picard inclusion holds on that closed interval.

## 2. What remains for G1

- A `Z` bound small enough for `C = 2 + delta` on a compact `(V,h)`
  rectangle (the slow-line holomorphic bound is
  [HILBERT16-CANCELLED-N.md](HILBERT16-CANCELLED-N.md); this majorant
  is finite but fat).
- Height mixing `|g_h| = O(epsilon^2)` at `C != 0` is
  [HILBERT16-HEIGHT-MIX.md](HILBERT16-HEIGHT-MIX.md).
- `T - h = O(epsilon)` along the actual `(V,h)` orbit.
- Height-section first-hit of the selected large first-root section
  (the transversal event).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16KillZeta.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16KillZeta.lean).
Python: `omnibias.dynamics.kill_zeta`.

## 3. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_kill_zeta.py -q
uv run --no-sync python -m benchmarks.hilbert16_kill_zeta
lake build OmnibiasAnalytic.Dynamics.Hilbert16KillZeta
```
