# Hilbert XVI: kill-line Stage-C C=2 integrating factor

This companion continues [the Stage-C C=0 envelope](HILBERT16-STAGE-C-ENV.md)
and [the first-root note](HILBERT16-ROOT-SADDLE.md) §5. On `lambda1 = -2`
Stage C starts at `y_1 = 1`, so `h_1 = eps^3`. The written comparison

    T_h <= C + C eps T/h + C eps^3/h

uses the integrating factor `h^(-C eps)`. At `C = 2` and `hmax = 1`
the exponent is `6 eps log(1/eps)`. The public `3 eps log(1/eps)`
majorant `6 (sqrt(eps)-eps)` from
[the post-corridor note](HILBERT16-POST-CORRIDOR.md) therefore lifts
by `C = 2` to `12 (sqrt(eps)-eps)`. On `eps in (0, 1/16]` one has
`sqrt(eps) <= 1/4`, so the exponent is at most `3` and
`(h/h_1)^{C eps} < 32`.

This is a sealed Stage-C C=2 integrating-factor bound on the kill line,
not `T(h) <= C (eps^2+h)` after the remaining integral, not first-hit,
C2, `dx_e` off the kill line, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- A height-section first-hit of the Stage-C `T(h)` orbit
  (the C=2 majorant is [the Stage-C T(h) note](HILBERT16-STAGE-C-INT.md)).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCIf.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCIf.lean).
Python: `omnibias.dynamics.stage_c_if`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_if.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_if
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCIf
```
