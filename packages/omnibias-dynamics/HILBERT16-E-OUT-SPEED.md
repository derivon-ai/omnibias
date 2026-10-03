# Hilbert XVI: kill-line comparison speed bound

This companion continues [shrinking-`eps` `E_out`](HILBERT16-E-OUT-EPS.md).
On the kill line `L = 0`, `lambda1 = -2`, `nu = eps`, the cubic field with
`h >= 4 eps^3` obeys `Vdot <= F(V, eps) := f(V) + 4 eps^3 g(V)`. The
`V`-derivative is `F_V = eps phi` with `phi = V^2 - 2 V - 2 eps + 4 eps^3`.
On `V in [-rho, -eps]`, `phi` is decreasing in `V` and its right-end value
is `eps^2 (1 + 4 eps) > 0`, so `F_V > 0`, `F` is increasing, and

    F(V, eps) <= F(-eps, eps) = - (3 eps^3 + (13/3) eps^4 + 4 eps^5) < 0.

Hence `-Vdot >= 3 eps^3` and the comparison hitting time of `V = -rho` is
at most `(rho - eps) / (3 eps^3)`, uniformly for every `eps in (0, 1/16]`.
This is an `O(1/eps^3)` comparison majorant, not a Lohner first-hit for
every `eps`, not GRAZING `E_sigma`, not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Certified GRAZING `E_sigma` first-hit on the `V ≈ 1` chart
  (the incoming comparison is [HILBERT16-E-SIGMA-SPEED.md](HILBERT16-E-SIGMA-SPEED.md)).
- A uniform Lohner first-hit for every `eps` (this note is an
  `O(1/eps^3)` comparison time, not `certify_stopped_event` on every
  square).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16EOutSpeed.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16EOutSpeed.lean).
Python: `omnibias.dynamics.e_out_speed`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_out_speed.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_out_speed
lake build OmnibiasAnalytic.Dynamics.Hilbert16EOutSpeed
```
