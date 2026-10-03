# Hilbert XVI: kill-line Stage-C C=2 `T-h` bootstrap

This companion continues [the Stage-C C=2 tight `T(h)` ratio](HILBERT16-STAGE-C-K.md)
and [the first-root note](HILBERT16-ROOT-SADDLE.md) §5. On `lambda1 = -2`
the sealed sandwich is `T <= 6 (eps^2+h)`. Along the comparison,

    |(T-h)_h| <= C K eps + 3 nu + C(1+K) eps^3 / h

with `C=2` and `nu=eps`, so the linear coefficient is `15 eps` and
the log coefficient is `42 eps^3 log(1/eps)` after integrating from
`h_1=eps^3` to `hmax=1`. Adding the sealed start gap, Interval
wrapping at the compact edge still yields `T-h < 1`. This is a
fixed-margin bound, not `O(eps)`, not first-hit, C2, `dx_e` off the
kill line, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- A height-section first-hit of the Stage-C `T(h)` orbit
  (the continuation rectangle is [the Stage-C rectangle note](HILBERT16-STAGE-C-RECT.md);
  the selected large section remains).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCBoot.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCBoot.lean).
Python: `omnibias.dynamics.stage_c_boot`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_boot.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_boot
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCBoot
```
