# Hilbert XVI: cancelled-N holomorphic Z on the kill compact

This companion continues [the rectangular kill majorant](HILBERT16-KILL-ZETA.md).
The cubic pieces of

    N = Vdot/eps - eps^2 L - eps lambda1 V + V^2 - V^3 / 3

cancel at `nu = 0`. Enclosing `N` on a polydisc therefore wraps an
`O(1)` cancellation and is not small enough for `C = 2 + delta`.
Factoring that cancellation gives holomorphic formulas for `Z` with
no division by `nu`. On the declared real compact

    nu in [0, 0.02],  v in [-0.5, 1.5],  L in [0, 1],  lambda1 = -2

Picard inclusion holds around `k0 = 3 v0 / l`, `|Z|` is `O(1)`, and

    2 eps |V| |Z| < 1.

That is a usable `C = 2 + delta` prefactor **on the slow line**. It is
not `T-h` along the actual `(V,h)` orbit, not `C != 0` `|g_h|`, not
height-section first-hit, G1, or Hilbert XVI. A strictly larger `nu`
box (`nu <= 0.2`) refuses Picard inclusion; the compact is declared,
not arbitrary.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What cancels

The unfolding slow line differs from the cubic `slow0` by
`f0 + f1 (v-v0)`. The lambda source then reduces to
`- nu^2 k lambda1 (v-v0)^2` once `v0` is the slow-line root, and the
`L` source reduces to `-4 nu^4 k^2 L (v-v0)^2 / l^2`. The remaining
`lambda=0` cubic quotient is holomorphic at `(nu, V) = (0, 0)`.

## 2. What remains for G1

- `T - h = O(epsilon)` along the actual `(V,h)` orbit.
- Height-section first-hit of the selected large first-root section
  (the transversal event).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

The `C != 0` mixing `|g_h| = O(nu^2)` is
[HILBERT16-HEIGHT-MIX.md](HILBERT16-HEIGHT-MIX.md).

Lean: [Hilbert16CancelledN.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16CancelledN.lean).
Python: `omnibias.dynamics.cancelled_n`.

## 3. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_cancelled_n.py -q
uv run --no-sync python -m benchmarks.hilbert16_cancelled_n
lake build OmnibiasAnalytic.Dynamics.Hilbert16CancelledN
```
