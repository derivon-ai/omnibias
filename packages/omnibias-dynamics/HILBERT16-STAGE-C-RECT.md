# Hilbert XVI: kill-line Stage-C continuation rectangle

This companion continues [the Stage-C `T-h` bootstrap](HILBERT16-STAGE-C-BOOT.md)
and [the first-root note](HILBERT16-ROOT-SADDLE.md) §5. On `lambda1 = -2`
the sealed `T-h < 1` bound gives `T < h+1`, so `|V| = sqrt(2T) < sqrt(2h+2)`.
At `hmax = 1` that left wall is `sqrt(4) = 2`. The matching-chart floor
`|V| = eps x` with `x >= 1/4` gives the right wall `eps/4 = 1/64` at
the compact edge. Interval wrapping still keeps the left wall `< 3`
and the right wall `> 0`. The comparison orbit therefore stays in

    V in [-2, -1/64],  h in [h_1, 1]

on this compact. This is a sealed continuation rectangle, not first-hit
of the selected large section, C2, `dx_e` off the kill line, G1, or
Hilbert XVI. The comparison first-hit of `{h=1}` is
[HILBERT16-STAGE-C-HIT.md](HILBERT16-STAGE-C-HIT.md).

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- A height-section first-hit of the Stage-C `T(h)` orbit
  (the continuation rectangle is this note; the comparison first-hit
  of `{h=1}` is [HILBERT16-STAGE-C-HIT.md](HILBERT16-STAGE-C-HIT.md);
  the selected large signed-label section remains).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCRect.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCRect.lean).
Python: `omnibias.dynamics.stage_c_rect`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_rect.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_rect
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCRect
```
