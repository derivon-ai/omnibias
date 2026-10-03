# Hilbert XVI: comparison-bootstrap integral of `T-h`

This companion continues [the pointwise `T_h` gap](HILBERT16-ORBIT-TH.md).
Along an actual outgoing branch,

    (T-h)_h = q/h + k - 1.

On a compact where `T <= K (eps^2 + h)`, `|q| <= C eps (eps^2 + T)`, and
`|k-1| <= 3 nu`, the slope splits exactly as

    C eps (eps^2 + K (eps^2 + h)) / h + 3 nu
        = C K eps + 3 nu + C (1+K) eps^3 / h.

The linear piece integrates to `O(eps)`. The `eps^3 / h` piece
integrates to `O(eps^3 log(hmax/h_e))`, majorized by
`log u <= 2 (sqrt(u) - 1)`. On the declared compact `eps = 1/n` for
square `n >= 16`, `y0 = 4`, `hmax = 1`, `x_* = 1`, `C = K = 2`,
`nu = eps`, the resulting majorant of `T-h` is `< 9 eps`.

This is the comparison bootstrap, not a Lohner-validated actual
`(V,h)` orbit, not height-section first-hit, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- A Lohner-validated actual `(V,h)` orbit (the cubic truncation is
  [HILBERT16-VH-ORBIT.md](HILBERT16-VH-ORBIT.md); this note is the
  comparison majorant).
- Uniform first-hit as `eps -> 0` (matching-chart `E_out` on
  `L in {9/25, 1/16, 0}` at one `eps` is [HILBERT16-E-OUT-SECTION.md](HILBERT16-E-OUT-SECTION.md);
  GRAZING `E_sigma` on `V ≈ 1` remains).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16ThIntegral.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ThIntegral.lean).
Python: `omnibias.dynamics.th_integral`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_th_integral.py -q
uv run --no-sync python -m benchmarks.hilbert16_th_integral
lake build OmnibiasAnalytic.Dynamics.Hilbert16ThIntegral
```
