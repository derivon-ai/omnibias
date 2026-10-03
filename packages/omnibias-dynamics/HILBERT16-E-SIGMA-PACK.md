# Hilbert XVI: `L in {9/25, 1/16}` wall-span GRAZING `E_sigma` cover

This companion continues [L=0 whole-wall span](HILBERT16-E-SIGMA-SPAN.md).
The GRAZING reverse cubic from `V=0`, `h=4 eps^3` has a certified
first-hit of `V=1/4`. On `L in {9/25, 1/16}` those return `h`-boxes
sit inside the declared rational span `[17/1000, 7/200]`. Split into
eighteen equal slabs of width `1/1000`, `certify_stopped_event` hits
GRAZING `E_sigma` uniquely and transversely on every slab at both
`L`. A single slab over the whole span is unresolved. The GRAZING
start `V=0` still excludes `E_sigma` on this compact horizon.

This is enclosure continuation of a declared span containing the
remaining L-pack wall boxes, not a single Lohner run from `V=0`,
not uniform in `eps`, not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Lohner `E_sigma` from the GRAZING start `V=0` in one run (this
  note restarts at `V=1/4` on a positive-width `h`-span).
- A uniform Lohner first-hit for every `eps` (a finite aligned
  shrinking pack is [HILBERT16-E-SIGMA-EPS.md](HILBERT16-E-SIGMA-EPS.md)).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16ESigmaPack.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaPack.lean).
Python: `omnibias.dynamics.e_sigma_pack`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_sigma_pack.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_sigma_pack
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaPack
```
