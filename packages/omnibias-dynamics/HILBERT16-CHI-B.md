# Hilbert XVI: kill-line `chi_b` threshold

This companion continues [the Stage-A wall](HILBERT16-STAGE-A.md) and
[coalescing capture](HILBERT16-COALESCING-CAPTURE.md) §5.2.
On `lambda1 = -2` the slow-line pre-rectangle time is

    S_pre = integral_0^a x / B_-(x) dx.

The log-sep pieces of the antiderivative leave

    sep * S_pre = sep ln sep + r1 ln r1 - r2 ln r2
                  + r2 ln(1+theta) - r1 ln theta

with `theta = 1/8`. The `sep -> 0` limit is `ln 9`. Interval arithmetic
on two slabs covering `sep in [1/2^16, 1]` encloses `sep * S_pre < 3`,
so `S_pre <= 3/sep`. Then `kappa_b = 4/sep` and

    chi_b = (sep / r1) * kappa_b = 4 / r1 < 9

since `r1 >= 1/2`. The chi-rectangle decay constant is `c = 1/16`.
This is a sealed `chi` threshold on a declared compact, not
`dx_e / d kappa`, not Stage C, first-hit, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- The uniform-in-`chi` bound `dx_e / d kappa <= C sep^2 exp(-c chi)`
  is [HILBERT16-DX-E-UNIF.md](HILBERT16-DX-E-UNIF.md).
  Leading factors are [HILBERT16-DX-E-LEADING.md](HILBERT16-DX-E-LEADING.md).
- Stage C first-root outgoing with uniform `a_min` is
  [HILBERT16-STAGE-C.md](HILBERT16-STAGE-C.md).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- The remaining `sep in (0, 1/2^16)` is a different (super-small) chart.

Lean: [Hilbert16ChiB.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ChiB.lean).
Python: `omnibias.dynamics.chi_b`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_chi_b.py -q
uv run --no-sync python -m benchmarks.hilbert16_chi_b
lake build OmnibiasAnalytic.Dynamics.Hilbert16ChiB
```
