# Hilbert XVI: comparison GRAZING `E_sigma` from `V=0`

This companion continues [declared-point `E_sigma`](HILBERT16-E-SIGMA-HIT.md).
On the kill-line reverse cubic (`L >= 0`, `lambda1 = -2`, `nu = eps`),
`Vdot_rev = -f + h(1+nu-nu V)`. For `V in [0, 6/5]`, `f <= 0` and
`1+nu-nu V > 0`, so `dh/dV <= V/(1+nu-nu V)`. The integral is exact;
`log(1+x)` is majorized by the cubic Taylor remainder at
`x = 6/79` (`V* = 6/5`, `eps = 1/16`). The resulting height majorant
makes `E_sigma(V*, h_up) > 0` while `E_sigma(0, 4 eps^3) < 0`. Along
the tube, `dE/dV >= 55/79 > 0`, so `E_sigma` has a unique increasing
zero on `L in {9/25, 1/16, 0}`. `V* = 1` does not yet change sign.
Lohner wrapping still refuses `certify_stopped_event` from `V = 0`.

This is a comparison first-hit from the GRAZING start, not a Lohner
event, not uniform in `eps`, not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Lohner `E_sigma` from the GRAZING start `V = 0` (this note is a
  comparison tube, not `certify_stopped_event`).
- A uniform Lohner first-hit for every `eps` (the comparison cover
  is [HILBERT16-E-SIGMA-UNIF.md](HILBERT16-E-SIGMA-UNIF.md)).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16ESigmaFrom0.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaFrom0.lean).
Python: `omnibias.dynamics.e_sigma_from0`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_sigma_from0.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_sigma_from0
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaFrom0
```
