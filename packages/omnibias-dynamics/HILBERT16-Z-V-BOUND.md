# Hilbert XVI: holomorphic `Z_v` bound on the kill compact

This companion continues [the unfrozen `Z_x` identities](HILBERT16-Z-X-GAP.md)
and [the cancelled-N holomorphic `Z`](HILBERT16-CANCELLED-N.md). On the
`lambda=0` slice, `Z0 = ell0 q1 / (9 v0^2 wall^3)` is rational in
`(nu, v, v0)`. Termwise `q1_v`, the quotient rule for `Z0`, and the
product rule for `d/dv[(wall+ell0)/wall^3]` are exact. Interval
arithmetic on the cancelled-N compact

    nu in [0, 0.02],  v in [-0.5, 1.5],  L in [0, 1],  lambda1 = -2

encloses `|Z_v| < 1/4` with Picard-included `k`, and the box excludes 0.

This is a holomorphic-chart bound in the slow-line coordinate `v`, not a
fold `Z_x` bound, not `sep > 0`, not first-hit, G1, or Hilbert XVI.

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

Lean: [Hilbert16ZVBound.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ZVBound.lean).
Python: `omnibias.dynamics.z_v_bound`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_z_v_bound.py -q
uv run --no-sync python -m benchmarks.hilbert16_z_v_bound
lake build OmnibiasAnalytic.Dynamics.Hilbert16ZVBound
```
