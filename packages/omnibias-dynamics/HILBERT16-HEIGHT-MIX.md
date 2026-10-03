# Hilbert XVI: C≠0 height mixing `|g_h| = O(nu^2)`

This companion continues [the cancelled-N bound](HILBERT16-CANCELLED-N.md).
On the normal chart the exact polynomials are

    ell = 1 + 2 nu v + C nu^2 h,
    V   = 1 - v - nu v^2 - C nu^2 v h.

Differentiating gives `V_v + ell = 0` and `V_h + C nu^2 v = 0`, hence
`v_V = -1/ell` and `v_h = -C nu^2 v / ell`. The first-order jet

    Vdot = -h + nu { V^3 - 3 V^2 + (V-1) h }

has `g = -1 + nu (V-1)`, independent of `h`. The C-term in `ell` and `V`
is `O(nu^2)`. On the declared compact

    nu <= 1/8,  |C| <= 3,  |v| <= 2,  |h| <= 2

one has `ell > 0` and `|ell_h| = |C| nu^2 <= 3 nu^2`, so
`|g_h| = O(nu^2)`. A box with `nu <= 1` refuses `ell > 0`.

This is not `T-h` along the actual `(V,h)` orbit, not height-section
first-hit, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- `T - h = O(epsilon)` along the actual `(V,h)` orbit
  (the pointwise `T_h` gap is [HILBERT16-ORBIT-TH.md](HILBERT16-ORBIT-TH.md);
  the comparison-bootstrap integral is [HILBERT16-TH-INTEGRAL.md](HILBERT16-TH-INTEGRAL.md)).
- Height-section first-hit of the selected large first-root section
  (the transversal event).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16HeightMix.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16HeightMix.lean).
Python: `omnibias.dynamics.height_mix`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_height_mix.py -q
uv run --no-sync python -m benchmarks.hilbert16_height_mix
lake build OmnibiasAnalytic.Dynamics.Hilbert16HeightMix
```
