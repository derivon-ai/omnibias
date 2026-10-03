# Hilbert XVI: kill-line Stage-A `dx_e / d kappa` leading factors

This companion continues [the Stage-A wall](HILBERT16-STAGE-A.md) and
[the `chi_b` threshold](HILBERT16-CHI-B.md). On `lambda1 = -2` the
slow-line event derivative factors as

    dx_e / d kappa = (A / x_e) exp(Psi_pre)
                     * exp integral eps (B' + y k_x) d tau.

The algebraic prefactor is `theta(1+theta) sep^2 / x_e`. With
`theta = 1/8` and `x_e >= a > 1/4` this is below `(1/2) sep^2`. The
integrand is at most `-sep (1-2 theta)/2 = -3 sep/8` and `X <= 2`, so
the tau-coefficient is `3 sep/16`. At the `chi_b` threshold
`kappa = 4/sep` the net exponent `(3/16)(4 - sep S_pre)` stays above
`1/8` after the `y0 = (1/16) sep^2` log remainder on
`sep in [1/2^16, 1]`, `eps in [0, 1/16]`.

This is a sealed leading-factor enclosure, not the uniform-in-`chi`
bound `C sep^2 exp(-c chi)`, not Stage C, first-hit, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- The uniform-in-`chi` bound `dx_e / d kappa <= C sep^2 exp(-c chi)`
  for every `chi >= chi_b` is [HILBERT16-DX-E-UNIF.md](HILBERT16-DX-E-UNIF.md).
- Stage C first-root outgoing with uniform `a_min` is
  [HILBERT16-STAGE-C.md](HILBERT16-STAGE-C.md).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.

Lean: [Hilbert16DxELeading.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16DxELeading.lean).
Python: `omnibias.dynamics.dx_e_leading`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_dx_e_leading.py -q
uv run --no-sync python -m benchmarks.hilbert16_dx_e_leading
lake build OmnibiasAnalytic.Dynamics.Hilbert16DxELeading
```
