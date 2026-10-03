# Hilbert XVI: outgoing x-corridor on the shrinking-root chart

This companion continues [the shrinking-root note](HILBERT16-SHRINKING-ROOT.md).
The inner rescaling `x = r1 xi` attracts to `xi = 1`. That is not outgoing
continuation to a physical section at fixed `x_*`. The two-root slow-line
map from the matching interface `x = r1 (1+theta)` to a compact
`x_* in (0, r2)` has a **bounded** increment of the cleared
antiderivative

    J(x) = -r1 log|x-r1| + r2 log|x-r2|,

because `dJ/dx = (r2-r1) x / ((x-r1)(x-r2))` exactly and the interface
term `r1 log r1` is majorized by `2 sqrt(r1) - 2 r1 -> 0` on `(0, 1]`.

The first-root wall `a = r1 - d` is negative once `r1 < d`. This is not
first-hit of the large **height** section, not a uniform `a_min`, not G1,
and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. Inner attractor vs outgoing corridor

On `x = r1 xi`, the rescaled slow ODE tends to `xi' = r2 (1 - xi)`,
whose attractor is the first root `xi = 1`. A large physical section at
fixed `x_*` requires `xi = x_*/r1 -> infinity`. That is a different
chart. Matching at `xi = 1+theta` (fixed `theta in (0,1)`) and integrating
in physical `x` up to `x_*` is the outgoing corridor.

## 2. Cleared I-map

Partial fractions give

    x / ((x-r1)(x-r2)) = (r1/(r1-r2))/(x-r1) + (r2/(r2-r1))/(x-r2).

Clearing the factor `r2-r1` produces `J` above. The identity

    -r1 (x-r2) + r2 (x-r1) - (r2-r1) x = 0

is the numerator of `J'`. From `x_lo = r1(1+theta)` the `r1 log r1`
remainder is bounded by `2 sqrt(r1)-2 r1`, which vanishes along
`r1 = 1/n^2`. The slow time to `x_*` therefore stays bounded as
`r1 -> 0`.

## 3. What remains for G1

- Height-section first-hit of the selected large first-root section
  (the `(V,h)` orbit; [post-corridor](HILBERT16-POST-CORRIDOR.md)
  restores `T_e = Theta(eps^2)`; the
  [alpha-0 envelope](HILBERT16-HEIGHT-ENVELOPE.md) conserves `T-h` on
  the comparison ODE, not the event).
- Uniform `a_min` (the wall `a = r1-d` fails for each fixed `d`).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16OutgoingCorridor.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16OutgoingCorridor.lean).
Python: `omnibias.dynamics.outgoing_corridor`.

## 4. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_outgoing_corridor.py -q
uv run --no-sync python -m benchmarks.hilbert16_outgoing_corridor
lake build OmnibiasAnalytic.Dynamics.Hilbert16OutgoingCorridor
```
