# Hilbert XVI: `dx_e` for every `lambda1 <= -2`

This companion continues the slab
[HILBERT16-DX-E-OFF.md](HILBERT16-DX-E-OFF.md). Residence in the
rectangle is at least `log(y0 / y_e) / X` with `X <= 2 rstar` and
`rstar = -lambda1 / 2`. The integrand bound `-3 sep/8` therefore
gives the tau-coefficient

    3 sep / (16 rstar).

For every `rstar >= 1` and every `sep` in `(0, 1]`,

    r1 >= 1/2,    u = sep/r1 <= 2,    a >= 3/8,

so the sealed ratio `h(u) < 11/5` still yields
`sep * S_pre < r1*(11/5)`. The threshold

    kappa * sep = r1*(11/5) + rstar

leaves the main exponent `3/16`. The log remainder carries the
factor `1/rstar`, so it is most negative at the kill edge
`rstar = 1`. The net stays above `1/8`, `chi_b <= 21/5`, and
`C < 2`. Dropping the `rstar` surplus stalls.

Every `lambda1 < -4` is included. Every `lambda1` in `[-3/2, -2)`
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

Lean: [Hilbert16DxERay.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16DxERay.lean).
Python: `omnibias.dynamics.dx_e_ray`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_dx_e_ray.py -q
uv run --no-sync python -m benchmarks.hilbert16_dx_e_ray
lake build OmnibiasAnalytic.Dynamics.Hilbert16DxERay
```
