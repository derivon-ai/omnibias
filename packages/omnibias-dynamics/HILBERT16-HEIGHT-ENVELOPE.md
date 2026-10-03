# Hilbert XVI: alpha-0 T-h envelope on chart O

This companion continues [the post-corridor matching](HILBERT16-POST-CORRIDOR.md).
The first-root height continuation in [the root-saddle note](HILBERT16-ROOT-SADDLE.md)
§5 uses `T_h = q/h + k` and the comparison

    T_h = 1 + C epsilon T/h + C epsilon^3 / h.

At `C = 0` this is `T_h = 1`, so `T - h` is conserved and equals the
exit gap

    T_e - h_e = epsilon^2 x_*^2 / 2 - epsilon^3 y0.

For `epsilon y0 < x_*^2 / 2` the gap is positive: the orbit starts
above the parabola `T = h` (`V = -sqrt(2h)`). That gap is `Theta(epsilon^2)`,
hence below any fixed margin for small `epsilon`. The AM-GM identity

    epsilon (epsilon^2 + w^2) - 2 epsilon^2 w = epsilon (epsilon - w)^2

is the pointwise ingredient of `|q| <= C epsilon (epsilon^2 + T)`. At
the matching section the exact leading ratio

    |q| / (epsilon (epsilon^2 + T)) = (x - r1)(r2 - x) / (1 + x^2 / 2)

is independent of `epsilon` and tends to `x (r2 - x) / (1 + x^2 / 2)` as
`r1 -> 0`, so a uniform `C` exists. The `C epsilon` perturbation is the
already-majorized factor `(h / h_e)^{C epsilon} -> 1`.

This is not first-hit of the large height section: the actual field is
not the comparison ODE, and the transversal event remains unsealed. Not
G1 and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. Conserved gap vs fixed margin

On the `C = 0` comparison, `T(h) = T_e + h - h_e`. At a compact
`hmax = O(1)` the gap `T - h = T_e - h_e` is independent of `h` and of
`r1`. The negative-`V` wall used by the written bootstrap sits a fixed
margin outside `V = -sqrt(2h)`; a `Theta(epsilon^2)` gap cannot reach it.

## 2. Uniform |q| ratio as r1 -> 0

The ratio `(x - r1)(r2 - x) / (1 + x^2 / 2)` increases toward
`x (r2 - x) / (1 + x^2 / 2)` along `r1 = 1/n` at a fixed matching
`x_*`. On `lambda1 = -2` that same ratio is `< 2` for **every** `x`:
see [HILBERT16-Q-RATIO-C2.md](HILBERT16-Q-RATIO-C2.md).

## 3. What remains for G1

- Height-section first-hit of the selected large first-root section
  (the actual-field `T - h = O(epsilon)` bootstrap still needs a `Z`
  bound small enough for `C = 2 + delta`
  ([HILBERT16-CANCELLED-N.md](HILBERT16-CANCELLED-N.md) is holomorphic
  on the slow line;
  [HILBERT16-KILL-ZETA.md](HILBERT16-KILL-ZETA.md) is a finite
  rectangular majorant including `L=0`), `C != 0` `|g_h|`
  ([HILBERT16-HEIGHT-MIX.md](HILBERT16-HEIGHT-MIX.md)), and the
  transversal event; the leading `|q|` ratio is
  [HILBERT16-Q-RATIO-C2.md](HILBERT16-Q-RATIO-C2.md); the `C=0` `k`
  jet is [HILBERT16-K-ZETA-REMAINDER.md](HILBERT16-K-ZETA-REMAINDER.md)).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16HeightEnvelope.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16HeightEnvelope.lean).
Python: `omnibias.dynamics.height_envelope`.

## 4. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_height_envelope.py -q
uv run --no-sync python -m benchmarks.hilbert16_height_envelope
lake build OmnibiasAnalytic.Dynamics.Hilbert16HeightEnvelope
```
