# Hilbert XVI: kill-line Stage-C C=0 `T(h)` envelope

This companion continues [the Stage-C start gap](HILBERT16-STAGE-C-GAP.md)
and [the first-root note](HILBERT16-ROOT-SADDLE.md) §5. On `lambda1 = -2`
the `C = 0` comparison is `T_h = 1`, so

    T(h) = T_e + h - h_1

with `h_1 = eps^3` at the Stage-C start `y_1 = 1`. The sealed start gap
gives `(T_e - h_1)/eps^2 > 1/32`, and Stage-B exit energy has
`T_e/eps^2 < 1`. Therefore

    (1/32) (eps^2 + h) <= T(h) <= eps^2 + h

on the `C = 0` comparison, uniformly in height. This is the written
`c (eps^2+h) <= T <= C (eps^2+h)` sandwich at `C = 0`, not a `C != 0`
integrating-factor orbit, first-hit, C2, `dx_e` off the kill line, G1,
or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- The remaining integral in the C≠0 Stage-C `T(h)` majorant / height-section first-hit
  (the C=2 integrating factor is [the Stage-C integrating-factor note](HILBERT16-STAGE-C-IF.md)).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCEnv.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCEnv.lean).
Python: `omnibias.dynamics.stage_c_env`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_env.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_env
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCEnv
```
