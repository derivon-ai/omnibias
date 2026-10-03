# Hilbert XVI: post-corridor (V,h) hypotheses on chart O

This companion continues [the outgoing x-corridor](HILBERT16-OUTGOING-CORRIDOR.md).
The first-root wall `a = r1 - d` is negative once `r1 < d`. After the
x-corridor the matching section sits at a compact `x_* in (r1, r2)`:

    V_* = -epsilon x_*,     T_* = epsilon^2 x_*^2 / 2,     h_e = epsilon^3 y0.

Then `|V_*| / epsilon = x_*` and `T_* = Theta(epsilon^2)` do **not**
vanish with `r1`. The leading slow-line cubic at `V = -epsilon x` is

    q / epsilon^3 = L + lambda1 x + x^2 = (x - r1)(x - r2),

so `T_h = q/h + k` has leading value `1 + (x-r1)(x-r2)/y0` at `h_e`.
A fixed `y0 > -2 (x_*-r1)(x_*-r2)` yields `T_h > 1/2`, independently of
`r1`. The continuation factor `(h / h_e)^{C epsilon}` has logarithm
`C epsilon log(h/y0) + 3 C epsilon log(1/epsilon)`; the second term is
majorized by `6 (sqrt(epsilon) - epsilon) -> 0` along `epsilon = 1/n`.

This restores the *hypotheses* of the written first-root height
continuation in [the root-saddle note](HILBERT16-ROOT-SADDLE.md) §5. It
is not first-hit of the large height section, not a sealed
`T - h = O(epsilon)` bootstrap, not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. Vanishing a_min is pre-corridor

The saddle rectangle used `|V| >= epsilon a_min` with `a_min ~ r1`. That
wall dies. The x-corridor moves the matching section to fixed `x_*`,
where `|V| = Theta(epsilon)` uniformly in `r1`. The saddle wall
`a = r1 - d` remains negative; the restored margin is `x_* - r1 > 0`,
not a uniform `a_min` at the colliding root.

## 2. Integrating factor

On a fixed height interval `h <= hmax` with `h_e = epsilon^3 y0`,

    (h / h_e)^{C epsilon} = exp(C epsilon log(h/y0) + 3 C epsilon log(1/epsilon)).

Even though `h / h_e -> infinity`, the exponent tends to 0. The factor
is uniformly bounded as `r1 -> 0` and as `epsilon -> 0`.

## 3. What remains for G1

- Height-section first-hit of the selected large first-root section
  (the `(V,h)` orbit from this matching section to the event; the
  `C=0` `T-h` envelope is
  [HILBERT16-HEIGHT-ENVELOPE.md](HILBERT16-HEIGHT-ENVELOPE.md); the
  `C=0` `k` jet is
  [HILBERT16-K-ZETA-REMAINDER.md](HILBERT16-K-ZETA-REMAINDER.md); the
  actual-field bootstrap is unsealed).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16PostCorridor.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16PostCorridor.lean).
Python: `omnibias.dynamics.post_corridor`.

## 4. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_post_corridor.py -q
uv run --no-sync python -m benchmarks.hilbert16_post_corridor
lake build OmnibiasAnalytic.Dynamics.Hilbert16PostCorridor
```
