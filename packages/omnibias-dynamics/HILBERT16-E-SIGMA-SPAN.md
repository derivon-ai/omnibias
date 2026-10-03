# Hilbert XVI: L=0 whole-wall `h`-span GRAZING `E_sigma` cover

This companion continues [wall-box `h`-interval cover](HILBERT16-E-SIGMA-BOX.md).
The GRAZING reverse cubic from `V=0`, `h=4 eps^3` has a certified
first-hit of `V=1/4`. On `L=0` that return `h`-box sits inside the
declared rational span `[19/1000, 1/25]`. Split into twenty-one
equal slabs of width `1/1000`, `certify_stopped_event` hits GRAZING
`E_sigma` uniquely and transversely on every slab. A single slab
over the whole span is unresolved. The GRAZING start `V=0` still
excludes `E_sigma` on this compact horizon.

This is enclosure continuation of a declared span containing the
whole `L=0` wall box, not the `L in {9/25, 1/16}` walls (those
extend below `19/1000`), not a single Lohner run from `V=0`, not
uniform in `eps`, not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Lohner `E_sigma` from the GRAZING start `V=0` in one run (this
  note restarts at `V=1/4` on a positive-width `h`-span).
- A uniform Lohner first-hit for every `eps`.
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16ESigmaSpan.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaSpan.lean).
Python: `omnibias.dynamics.e_sigma_span`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_sigma_span.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_sigma_span
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaSpan
```
