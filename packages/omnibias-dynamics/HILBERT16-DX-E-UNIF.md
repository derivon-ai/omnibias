# Hilbert XVI: kill-line uniform-in-`chi` `dx_e / d kappa`

This companion continues [the `dx_e` leading factors](HILBERT16-DX-E-LEADING.md)
and [the `chi_b` threshold](HILBERT16-CHI-B.md). On `lambda1 = -2` the
Stage-A event derivative on the `chi_b` compact satisfies

    dx_e / d kappa <= (1/2) sep^2 exp(-after(chi)).

For every `chi >= chi_b` the extra exponent is
`(3/16) r1 (chi - chi_b)`, and `(3/16) r1 >= 3/32`. Absorbing the
threshold floor then yields

    dx_e / d kappa <= C sep^2 exp(-(3/32) chi)

with Interval `C < 2`. The written decay `c = 1/16` is strictly
weaker by `1/32`. This is a sealed uniform-in-`chi` majorant on a
declared compact, not Stage C, first-hit, `dx_e` off the kill line,
G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Stage C first-root outgoing with uniform `a_min` is
  [HILBERT16-STAGE-C.md](HILBERT16-STAGE-C.md). The Stage-C exit
  energy is [HILBERT16-STAGE-C-EXIT.md](HILBERT16-STAGE-C-EXIT.md).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16DxEUnif.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16DxEUnif.lean).
Python: `omnibias.dynamics.dx_e_unif`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_dx_e_unif.py -q
uv run --no-sync python -m benchmarks.hilbert16_dx_e_unif
lake build OmnibiasAnalytic.Dynamics.Hilbert16DxEUnif
```
