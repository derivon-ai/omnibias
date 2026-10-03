# Hilbert XVI: kill-line Stage-C comparison first-hit of h=1

This companion continues [the Stage-C continuation rectangle](HILBERT16-STAGE-C-RECT.md)
and [the first-root note](HILBERT16-ROOT-SADDLE.md) §5. On `lambda1 = -2`
the sealed rectangle has `|V| >= 1/64`, so `hdot = -V h >= (1/64) h`.
Height is strictly increasing. From `h_1 = eps^3 = 1/4096 < 1` the
time to `hmax = 1` is at most `64 * 3 * log(1/eps) = 192 ln(16)` at
the compact edge, and Interval wrapping stays `< 1024`. This is a
unique comparison first-hit of `{h=1}`, not a Lohner orbit, not the
physical signed-label section, not chart O, C2, `dx_e` off the kill
line, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- First-hit of the selected large outgoing physical signed-label
  section inside the Stage-C rectangle (the comparison first-hit of
  `E_out` is [HILBERT16-STAGE-C-SEC.md](HILBERT16-STAGE-C-SEC.md);
  a Lohner orbit from Stage-C start remains).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCHit.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCHit.lean).
Python: `omnibias.dynamics.stage_c_hit`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_hit.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_hit
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCHit
```
