# Hilbert XVI: kill-line Stage-C start gap at `y_1 = 1`

This companion continues [the Stage-C leading `T_h` floor](HILBERT16-STAGE-C-TH.md)
and [the first-root note](HILBERT16-ROOT-SADDLE.md) §5. On `lambda1 = -2`
matching-chart Stage C starts at the sep-independent height `y_1 = 1`
after Stage B, so `h_1 = eps^3`. Then

    (T_e - h_1) / eps^2 = x_e^2 / 2 - eps.

At the written wall `x = 1/2` this is `1/8 - eps`, and on
`eps in [0, 1/16]` the exact edge value is `1/4096`. Interval wrapping
on the Stage-B end box still yields `(T_e - h_1)/eps^2 > 1/32`: the
orbit starts above `T = h` at the actual Stage C height, not the
Stage A `y0` used by [the exit-energy note](HILBERT16-STAGE-C-EXIT.md).

This is a sealed Stage-C start-gap enclosure on the kill line, not an
outgoing orbit, first-hit, C2, `dx_e` off the kill line, G1, or
Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- A `C != 0` Stage-C height-continuation orbit / height-section first-hit
  (the C=0 sandwich is [the Stage-C envelope note](HILBERT16-STAGE-C-ENV.md)).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCGap.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCGap.lean).
Python: `omnibias.dynamics.stage_c_gap`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_gap.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_gap
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCGap
```
