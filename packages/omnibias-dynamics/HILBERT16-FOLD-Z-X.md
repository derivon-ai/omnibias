# Hilbert XVI: matching-chart fold I-map `Z_x`

This companion continues [the slow-line `Z_V` chain](HILBERT16-Z-SLOW-V.md)
and [the unfrozen `Z_x` identities](HILBERT16-Z-X-GAP.md). On the fold wall
`r in [1.4, 1.6]`, holomorphic `Z` is a function of slow-line `v`. Matching
`V = -eps x` gives `v_x = eps / ell` with `ell = 1 + 2 nu v`, so

    Z_x = Z_v * eps / ell.

Interval arithmetic on the fold holomorphic `Z_v` box, `eps in [0, 0.02]`,
and `ell >= 49/50` encloses `|Z_x| < 1/100`. The extra C2 term
`eps^2 x^4 Z_x` is then `O(eps^3)` on this compact. At `eps = 0` the extra
term vanishes.

This is a fold I-map `Z_x` bound under the matching identification, not
`sep > 0`, not first-hit, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Stage A uniform-in-`chi` `dx_e / d kappa` is
  [HILBERT16-DX-E-UNIF.md](HILBERT16-DX-E-UNIF.md). Stage C
  `a_min` is [HILBERT16-STAGE-C.md](HILBERT16-STAGE-C.md).
  Kill-line Stage B is
  [HILBERT16-STAGE-B.md](HILBERT16-STAGE-B.md). Kill-line Stage A
  wall identities are [HILBERT16-STAGE-A.md](HILBERT16-STAGE-A.md).
  The `chi_b` threshold is [HILBERT16-CHI-B.md](HILBERT16-CHI-B.md).
  Leading `dx_e` factors are
  [HILBERT16-DX-E-LEADING.md](HILBERT16-DX-E-LEADING.md).
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16FoldZX.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16FoldZX.lean).
Python: `omnibias.dynamics.fold_z_x`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_fold_z_x.py -q
uv run --no-sync python -m benchmarks.hilbert16_fold_z_x
lake build OmnibiasAnalytic.Dynamics.Hilbert16FoldZX
```
