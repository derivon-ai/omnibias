# Hilbert XVI: kill-line Stage-C `a_min` floor

This companion continues [kill-line Stage B](HILBERT16-STAGE-B.md) and
[the first-root note](HILBERT16-ROOT-SADDLE.md) §5. On `lambda1 = -2`
the written Stage-C rectangle uses `a_min ~ rstar/2 = 1/2`. After
Stage-B Picard the end box stays inside the guess `[1/4, 4/3]`, so a
declared floor `a_min = 1/4` holds after Interval wrapping. The
integrating factor `1/x` is then at most `4` on the guess, and
matching-chart `|V| = eps x` stays at least `eps/4`.

This is a sealed Stage-C geometric floor on the kill line, not an
outgoing orbit, first-hit, C2, `dx_e` off the kill line, G1, or
Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- An actual Stage-C outgoing orbit / height-section first-hit is
  started by [HILBERT16-STAGE-C-EXIT.md](HILBERT16-STAGE-C-EXIT.md)
  and [HILBERT16-STAGE-C-TH.md](HILBERT16-STAGE-C-TH.md);
  the height-continuation orbit remains open.
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageC.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageC.lean).
Python: `omnibias.dynamics.stage_c`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageC
```
