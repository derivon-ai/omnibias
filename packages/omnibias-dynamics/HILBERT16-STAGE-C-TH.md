# Hilbert XVI: kill-line Stage-C leading `T_h` floor

This companion continues [the Stage-C exit energy](HILBERT16-STAGE-C-EXIT.md)
and [the first-root note](HILBERT16-ROOT-SADDLE.md) §5. On `lambda1 = -2`
the leading height derivative at matching-chart exit is

    T_h = 1 + (x - r1)(x - r2) / y_1.

Stage C starts at the sep-independent height `y_1 = 1` after Stage B.
At the midpoint `x = 1` the product is `-sep^2 / 4`, so the worst value
on `sep in [0, 1]` is `3/4`. Interval wrapping on the Stage-B end box
still yields `T_h > 1/2`.

This is a sealed Stage-C leading `T_h` floor on the kill line, not an
outgoing orbit, first-hit, C2, `dx_e` off the kill line, G1, or
Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- An actual Stage-C height-continuation orbit / height-section first-hit
  (the start gap at `y_1=1` is [the Stage-C start-gap note](HILBERT16-STAGE-C-GAP.md)).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCTh.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCTh.lean).
Python: `omnibias.dynamics.stage_c_th`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_th.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_th
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCTh
```
