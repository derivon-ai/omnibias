# Hilbert XVI: kill-line Stage-C C=2 `T(h)` majorant

This companion continues [the Stage-C C=2 integrating factor](HILBERT16-STAGE-C-IF.md)
and [the first-root note](HILBERT16-ROOT-SADDLE.md) §5. On `lambda1 = -2`
the comparison

    T_h <= C + C eps T/h + C eps^3/h

with `C = 2` and integrating factor `h^(-C eps)` integrates to

    T(h) <= (h/h_1)^{C eps} T_e + (C/(1-alpha)) h + (h/h_1)^{C eps} eps^2

where `alpha = C eps <= 1/8` on `eps in (0, 1/16]` and `h_1 = eps^3`,
`hmax = 1`. The sealed factor is `< 32` and `T_e/eps^2 < 1`, so the
`eps^2` coefficient is `< 64`. The slope `C/(1-alpha)` equals `16/7`
at the compact edge and Interval wrapping stays `< 3`. Therefore

    T(h) <= 64 (eps^2 + h)

on the C=2 comparison. This is a sealed Stage-C `T(h)` majorant, not
first-hit, C2, `dx_e` off the kill line, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- A height-section first-hit of the Stage-C `T(h)` orbit
  (the C=2 lower envelope is [the Stage-C lower note](HILBERT16-STAGE-C-LO.md)).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCInt.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCInt.lean).
Python: `omnibias.dynamics.stage_c_int`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_int.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_int
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCInt
```
