# Hilbert XVI: `dx_e` on `lambda1` in `[-4, -2]`

This companion continues the kill-line factors in
[HILBERT16-DX-E-LEADING.md](HILBERT16-DX-E-LEADING.md). For
`r2 = r1 + sep` the ratio `sep * S_pre / r1` collapses to a function
of `u = sep / r1` alone:

    h(u) = u ln u - (1+u) ln(1+u) + (1+u) ln(9/8) - ln(1/8).

Slabs on `u in (0, 2]` enclose `h(u) < 11/5`. On
`rstar = -lambda1/2` in `[1, 2]` and `sep` in `(0, 1]`,

    r1 >= 1/2,    u <= 2,    a = r1 - sep/8 >= 3/8,
    bnd <= rstar <= 2.

So `sep * S_pre < r1 * (11/5)`. The slope bound `-3 sep/8` and
`X <= 2` keep the tau-coefficient `3 sep/16`. The threshold
`kappa * sep = r1*(11/5) + rstar/2` leaves the net exponent above
`1/8` after the `y0` log remainder. Then `chi_b <= 21/5` and the
factored majorant has `C < 2`. Comparing `h` with `1` stalls.

`lambda1 = -2` is the edge `rstar = 1`. Every `lambda1` in
`[-4, -2)` is off that edge. Every `lambda1 <= -2` is
[dx_e_ray](HILBERT16-DX-E-RAY.md). Every `lambda1` in `[-3/2, -2)`
is [dx_e_near](HILBERT16-DX-E-NEAR.md). Every `lambda1` in
`(-3/2, 0)` is [dx_e_open](HILBERT16-DX-E-OPEN.md). This is not
Stage C, not first-hit, not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.

Lean: [Hilbert16DxEOff.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16DxEOff.lean).
Python: `omnibias.dynamics.dx_e_off`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_dx_e_off.py -q
uv run --no-sync python -m benchmarks.hilbert16_dx_e_off
lake build OmnibiasAnalytic.Dynamics.Hilbert16DxEOff
```
