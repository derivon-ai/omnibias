# Hilbert XVI: lower aligned parametric-`eps` GRAZING `E_sigma` cover

This companion continues [the `[1/25, 1/16]` three-slab cover](HILBERT16-E-SIGMA-EPS-SPAN.md).
On the reverse cubic with `eps` a `PolynomialFlow` parameter, the aligned
restart `(V, h) = (1/4, 1/40)` has unique transverse first-hit of GRAZING
`E_sigma` on six equal slabs of width `1/128` covering `[1/64, 1/16]` at
`L = 0`. That compact contains the previous `[1/25, 1/16]` span and
`{1/16, 1/20, 1/25, 1/32, 1/64}`. The last slab containing `eps = 1/16`
certifies on `L in {9/25, 1/16}`. A single slab over the whole compact
is unresolved. The GRAZING start `V = 0`, `h = 4 eps^3` still excludes
`E_sigma` on the last slab.

This is enclosure continuation of a declared lower `eps` compact from
the aligned restart, not a uniform-in-`eps` theorem for every `eps`, not
Lohner from `V = 0` on that compact, not `Z_x` C2, not G1, and not
Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- A uniform Lohner first-hit for every `eps` (this note is one compact
  `[1/64, 1/16]` from the aligned restart, not `(0, eps0]` and not from
  `V = 0` on that compact).
- Physical C2 with a `Z_x` bound and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16ESigmaEpsLo.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaEpsLo.lean).
Python: `omnibias.dynamics.e_sigma_eps_lo`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_sigma_eps_lo.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_sigma_eps_lo
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaEpsLo
```
