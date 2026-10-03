# Hilbert XVI: `dx_e` for `lambda1` in `[-3/2, -2)`

This companion continues the ray
[HILBERT16-DX-E-RAY.md](HILBERT16-DX-E-RAY.md). For
`rstar = -lambda1/2` in `[3/4, 1)` and every `sep` in `(0, 1]`,

    a = rstar - (5/8) sep >= 1/8,
    r1 >= 1/4,
    u = sep/r1 <= 4.

Slabs keep `h(u) < 11/5` on `(2, 4]`, and the sealed ray bound
covers `(0, 2]`. The threshold

    kappa * sep = r1*(11/5) + rstar

again leaves the main exponent `3/16`. Because `ln(1/16) < -2`, the
log remainder decreases in `sep` and is most negative at
`rstar = 3/4`, `sep = 1`. The net stays above `1/8`,
`chi_b <= 26/5`, the extra coefficient is at least `1/16`, and
`C < 2`. Dropping the `rstar` surplus stalls.

Every `lambda1` in `(-3/2, 0)` is
[dx_e_open](HILBERT16-DX-E-OPEN.md). This is not Stage C,
not first-hit, not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.

Lean: [Hilbert16DxENear.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16DxENear.lean).
Python: `omnibias.dynamics.dx_e_near`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_dx_e_near.py -q
uv run --no-sync python -m benchmarks.hilbert16_dx_e_near
lake build OmnibiasAnalytic.Dynamics.Hilbert16DxENear
```
