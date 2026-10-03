# Hilbert XVI: kill-line Stage-C C=2 lower `T(h)` envelope

This companion continues [the Stage-C C=2 `T(h)` majorant](HILBERT16-STAGE-C-INT.md)
and [the first-root note](HILBERT16-ROOT-SADDLE.md) §5. On `lambda1 = -2`
the sealed leading floor is `T_h > 1/2` at `y_1 = 1`. The worst midpoint
product `-sep^2/4` only gets weaker as `y` grows, so `T_h >= 1/2`
persists on the Stage-C rectangle. Integrating the comparison slope
`1/2` from `h_1 = eps^3` gives

    T(h) >= T_e + (1/2) (h - h_1).

The start remainder `T_e/eps^2 - (1/32)(1+eps)` is `47/512` at the
written wall and Interval wrapping stays `> 1/16`. The `h`-coefficient
`1/2 - 1/32 = 15/32` is positive, so the worst height is `h = h_1`.
Therefore

    T(h) >= (1/32) (eps^2 + h)

on the C=2 comparison, jointly with the sealed upper majorant
`T(h) <= 64 (eps^2+h)`. This is a sealed two-sided Stage-C energy
sandwich, not first-hit, C2, `dx_e` off the kill line, G1, or
Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- A height-section first-hit of the Stage-C `T(h)` orbit
  (the tight `K=6` sandwich is [the Stage-C ratio note](HILBERT16-STAGE-C-K.md);
  `T-h=O(eps)` and continuation to `hmax` remain).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCLo.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCLo.lean).
Python: `omnibias.dynamics.stage_c_lo`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_lo.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_lo
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCLo
```
