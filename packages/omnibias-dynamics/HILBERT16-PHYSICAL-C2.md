# Hilbert XVI: frozen-Z C2 remainder versus the lifted fold

This companion records the exact first-log-derivative and C2 gaps of
`log D'` when the actual slow-line field is the lift plus a remainder
`eps^2 x^3 Z` with `Z` frozen in `x`. The identities are exact. They
do **not** pass G1: `Z_x`, `sep > 0`, and a uniform-in-`eps` majorant
remain open.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. Frozen remainder

Write

    B = (x - r)^2 + mu + eps^2 x^3 Z,    dx / d kappa = B / x.

The linear-in-`eps` lift `mu = beta0 eps r^3` mismatches
`beta0 eps x^3` by `beta0 eps (x^3 - r^3)`. With `Z` independent of
`x`, implicit differentiation gives

    (log D')_kappa - (log D'_lift)_kappa = 2 eps^2 x Z,
    (log D')_kappa kappa - (log D'_lift)_kappa kappa = 2 eps^2 Z B / x.

## 2. What remains for G1

The rectangular Cauchy majorant on the fold compact is too pessimistic
to consume as a uniform-in-`eps` C2 bound. Differentiating `Z` in `x`
records an exact extra term
([HILBERT16-Z-X-GAP.md](HILBERT16-Z-X-GAP.md)). A matching-chart fold
I-map `|Z_x|<1/100` enclosure is
[HILBERT16-FOLD-Z-X.md](HILBERT16-FOLD-Z-X.md). A
holomorphic-chart `|Z_v|<1/4` enclosure on the kill compact is
[HILBERT16-Z-V-BOUND.md](HILBERT16-Z-V-BOUND.md). The
slow-line `Z_V` chain and fold holomorphic `Z_v` are
[HILBERT16-Z-SLOW-V.md](HILBERT16-Z-SLOW-V.md). The
`sep > 0` charts, height-section first-hit on chart O
([HILBERT16-HEIGHT-ENVELOPE.md](HILBERT16-HEIGHT-ENVELOPE.md) is the
`C=0` comparison, not the event), complete first-hit, and
Stage B/C continuation stay open.

Lean: [Hilbert16PhysicalC2.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16PhysicalC2.lean).
Python: `omnibias.dynamics.physical_c2`.

## 3. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_physical_c2.py -q
uv run --no-sync python -m benchmarks.hilbert16_physical_c2
lake build OmnibiasAnalytic.Dynamics.Hilbert16PhysicalC2
```
