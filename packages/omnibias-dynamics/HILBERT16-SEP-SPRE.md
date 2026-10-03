# Hilbert XVI: `sep * S_pre` on every `sep` in `(0, 1]`

This companion continues the compact chi threshold in
[HILBERT16-CHI-B](HILBERT16-CHI-B.md), whose Interval slabs stop at
`sep = 1/2^16`. On `lambda1 = -2`,

    sep * S_pre = sep ln sep + r1 ln r1 - r2 ln r2
                  + r2 ln(9/8) - r1 ln(1/8)

with `r1 = 1 - sep/2` and `r2 = 1 + sep/2`. As `sep -> 0`,
`sep ln sep -> 0` and the smooth part tends to `ln 9`. Forty-eight
dyadic slabs from `2^{-48}` to `1`, plus a tail on `(0, 2^{-48}]`
that uses `sep ln sep <= 0`, enclose

    sep * S_pre < 11/5 < 3.

Thus `chi_b = 4/r1 <= 8 < 9` for every `sep` in `(0, 1]`, since
`r1 >= 1/2`. The same slabs enclose the `dx_e` log remainder on
`eps` in `[0, 1/16]`. With `sep * S_pre < 11/5` the net exponent
stays above `1/8`. `ln(1/16) < -2`, so that remainder decreases in
`sep` on `(0, 1]`. Feeding `S = 4` into the net floor stalls.

This closes the kill-line hole `sep in (0, 1/2^16)` for `sep * S_pre`
and for the leading net exponent. It is not `dx_e` off the kill line
(`lambda1 != -2`), not a uniform-in-chi bound, not Stage C, not
first-hit, not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.

Lean: [Hilbert16SepSpre.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16SepSpre.lean).
Python: `omnibias.dynamics.sep_spre`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_sep_spre.py -q
uv run --no-sync python -m benchmarks.hilbert16_sep_spre
lake build OmnibiasAnalytic.Dynamics.Hilbert16SepSpre
```
