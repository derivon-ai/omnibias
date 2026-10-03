# Hilbert XVI: Cauchy majorant for Z on a fold compact

This companion seals a rectangular Cauchy majorant for the divided
remainder `Z` on a **declared** real compact of the fold wall
`lambda1^2 = 4 L`, with `L = r^2` and `lambda1 = -2 r`. The identities
are exact. The Picard box for implicit `k` is a contraction check, not
a physical remainder. The bound does **not** pass G1.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. Fold wall

On `sep = 0`,

    B_- = L + lambda1 x + x^2 = (x - r)^2

exactly when `L = r^2` and `lambda1 = -2 r`. The discriminant identity
`lambda1^2 - 4 L = 0` is the same wall.

## 2. Implicit `k` and Cauchy majorant

The analytic `k` equation

    k = 3 v0 / l + nu^2 k^2 lambda1 / l^2 + 4 nu^4 k^3 L / l^4

is a Picard map around `k0 = 3 v0 / l`. On the compact
`r in [1.4, 1.6]` and the polydisc `|V| <= 0.08`, `|nu| <= 0.02`, a
pad of radius `0.4` contains its image. Enclosing the holomorphic
numerator

    N = Vdot/eps - eps^2 L - eps lambda1 V + V^2 - V^3 / 3

then bounds `|Z| <= |N| / (R^3 E k_min)`. A real sample at
`nu = 1/64`, `r = 3/2`, `v = 1` lies below the bound. A strictly
larger box (`|nu| <= 1/32`, `r in [1.25, 1.75]`) refuses Picard
inclusion; the compact is declared, not arbitrary.

Physical C2 of `log D'` off the lifted map
([HILBERT16-PHYSICAL-C2.md](HILBERT16-PHYSICAL-C2.md)), height-section
first-hit on chart O
([HILBERT16-HEIGHT-ENVELOPE.md](HILBERT16-HEIGHT-ENVELOPE.md) is the
`C=0` `T-h` comparison), the kill-compact `Z` majorant
([HILBERT16-KILL-ZETA.md](HILBERT16-KILL-ZETA.md)), complete first-hit, and Stage
B/C continuation stay open.

Lean: [Hilbert16FoldZeta.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16FoldZeta.lean).
Python: `omnibias.dynamics.fold_zeta`.

## 3. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_fold_zeta.py -q
uv run --no-sync python -m benchmarks.hilbert16_fold_zeta
lake build OmnibiasAnalytic.Dynamics.Hilbert16FoldZeta
```
