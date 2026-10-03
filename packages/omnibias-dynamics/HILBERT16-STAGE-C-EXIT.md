# Hilbert XVI: kill-line Stage-C exit energy

This companion continues [the Stage-C `a_min` floor](HILBERT16-STAGE-C.md)
and [the first-root note](HILBERT16-ROOT-SADDLE.md) §5. On `lambda1 = -2`
matching-chart exit is `V = -eps x`, so

    T_e = eps^2 x_e^2 / 2,    h_e = eps^3 y0,

with `y0 <= mu = 1/16`. After Stage B the end box has `x >= 1/4`, and
`x^2 / 2` at the written wall `x = 1/2` is `1/8`. Interval wrapping
still gives `T_e / eps^2 > 1/16`. The product `eps y0` is at most
`1/256`, so the exit gap `T_e - h_e` stays positive: the orbit starts
above `T = h`.

This is a sealed Stage-C exit-energy enclosure on the kill line, not an
outgoing orbit, first-hit, C2, `dx_e` off the kill line, G1, or
Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- An actual Stage-C height-continuation orbit / height-section first-hit.
  Leading `T_h > 1/2` is
  [HILBERT16-STAGE-C-TH.md](HILBERT16-STAGE-C-TH.md).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCExit.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCExit.lean).
Python: `omnibias.dynamics.stage_c_exit`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_exit.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_exit
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCExit
```
