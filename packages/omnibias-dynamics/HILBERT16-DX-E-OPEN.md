# Hilbert XVI: `dx_e` for `lambda1` in `(-3/2, 0)`

This companion continues
[HILBERT16-DX-E-NEAR.md](HILBERT16-DX-E-NEAR.md). For
`rstar = -lambda1/2` in `(0, 3/4)` the rectangle exists on

    sep in (0, min(1, (8/5) rstar)),

where `a > 0` and `u = sep/r1 < 8`. Slabs keep `h(u) < 11/5` on
`(4, 8]`, and the sealed near bound covers `(0, 4]`. The threshold

    kappa * sep = r1*(11/5) + rstar

leaves the main exponent `3/16`. The log remainder is at most

    (3/10) eps (2 ln sep + ln(1/16)).

The cap

    eps <= (1/6) / -(2 ln sep + ln(1/16))

keeps that remainder above `-1/20`, so the net stays above
`11/80 > 1/8`. As `a -> 0`, `chi_b` approaches `36/5` and the extra
coefficient approaches `3/80`. The decay `1/32` is strictly weaker,
and

    C < (1/5) / a.

Holding `eps = 1/16` at `sep = 1/4096` stalls. `lambda1 <= -3/2` is
already sealed. This is not Stage C, not first-hit, not G1, and not
Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.

Lean: [Hilbert16DxEOpen.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16DxEOpen.lean).
Python: `omnibias.dynamics.dx_e_open`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_dx_e_open.py -q
uv run --no-sync python -m benchmarks.hilbert16_dx_e_open
lake build OmnibiasAnalytic.Dynamics.Hilbert16DxEOpen
```
