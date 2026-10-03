# Hilbert XVI: unfrozen-Z first-log-derivative gap including `Z_x`

This companion continues [the frozen-Z C2 identities](HILBERT16-PHYSICAL-C2.md).
Along the slow line, `dx/dkappa = B/x` with

    B = (x - r)^2 + mu + eps^2 x^3 Z.

If `Z` depends on `x`, implicit differentiation gives

    (x B_x - B) - (x B0_x - B0) = eps^2 (2 x^3 Z + x^4 Z_x).

When `Z_x = 0` this recovers the frozen gap `2 eps^2 x^3 Z`. The extra
term is `eps^2 x^4 Z_x`. The identities are exact. They do **not** bound
`Z_x`, do not give `sep > 0`, first-hit, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- `sep > 0` charts.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation.

Matching-chart fold I-map `Z_x` is
[HILBERT16-FOLD-Z-X.md](HILBERT16-FOLD-Z-X.md).

Lean: [Hilbert16ZXGap.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ZXGap.lean).
Python: `omnibias.dynamics.z_x_gap`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_z_x_gap.py -q
uv run --no-sync python -m benchmarks.hilbert16_z_x_gap
lake build OmnibiasAnalytic.Dynamics.Hilbert16ZXGap
```
