# Hilbert XVI: slow-line `Z_V` chain and fold holomorphic `Z_v`

This companion continues [the holomorphic `Z_v` bound](HILBERT16-Z-V-BOUND.md)
and [the unfrozen `Z_x` identities](HILBERT16-Z-X-GAP.md). On the slow line,
`V = 1 - v - nu v^2`, so `V_v = -ell` with `ell = 1 + 2 nu v`. The
holomorphic remainder in the `v`-chart therefore satisfies

    Z_V = Z_v / V_v = - Z_v / ell.

Interval arithmetic on the cancelled-N kill compact encloses `|Z_V| < 1/4`
and excludes 0. The same holomorphic `Z_v` formula on the fold wall

    r in [1.4, 1.6],  L = r^2,  lambda1 = -2 r

encloses `|Z_v| < 1/4` (and then `|Z_V| < 1/4`) with Picard-included `k`.
The rectangular fold Cauchy majorant does not.

This is a slow-line `V`-chart bound plus a holomorphic fold `Z_v` bound,
not a fold I-map `Z_x` bound, not `sep > 0`, not first-hit, G1, or
Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Matching-chart fold I-map `Z_x` is
  [HILBERT16-FOLD-Z-X.md](HILBERT16-FOLD-Z-X.md).
- `sep > 0` charts.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16ZSlowV.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ZSlowV.lean).
Python: `omnibias.dynamics.z_slow_v`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_z_slow_v.py -q
uv run --no-sync python -m benchmarks.hilbert16_z_slow_v
lake build OmnibiasAnalytic.Dynamics.Hilbert16ZSlowV
```
