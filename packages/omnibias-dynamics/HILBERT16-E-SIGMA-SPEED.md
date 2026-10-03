# Hilbert XVI: incoming GRAZING comparison speed bound

This companion continues [kill-line comparison speed](HILBERT16-E-OUT-SPEED.md).
On the kill line `L = 0`, `lambda1 = -2`, `nu = eps`, the reversed cubic
field starting at `V = 0`, `h = 4 eps^3` has `Vdot_rev >= -F(V, eps)` with
the same comparison polynomial `F = f + 4 eps^3 g`. On `V in [0, 1]`,
`phi(0) = -2 eps + 4 eps^3 < 0` for `eps in (0, 1/16]`, so `F_V < 0`,
`F` is decreasing, and

    F(V, eps) <= F(0, eps) = -4 eps^3 (1 + eps) < 0.

Hence `Vdot_rev >= 4 eps^3 (1 + eps)` while `h >= h(0)` and `g < 0`, and
the comparison time to cross `Delta V = 1` is at most
`1 / (4 eps^3 (1 + eps))`. Lohner wrapping refuses a certified `E_sigma`
first-hit on this compact. This is an `O(1/eps^3)` comparison majorant,
not GRAZING first-hit, not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Certified GRAZING `E_sigma` first-hit on the `V ≈ 1` chart (Lohner
  wrapping on the reversed cubic; this note is a comparison time;
  the incoming `V=1/4` wall is [HILBERT16-E-SIGMA-IN.md](HILBERT16-E-SIGMA-IN.md)).
- A uniform Lohner first-hit for every `eps` on the outgoing matching
  chart.
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16ESigmaSpeed.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaSpeed.lean).
Python: `omnibias.dynamics.e_sigma_speed`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_sigma_speed.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_sigma_speed
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaSpeed
```
