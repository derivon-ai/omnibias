# Hilbert XVI: tracked product and slow-line entry-exit leading map

This companion records the first-derivative repair of the super-small
kill sequence of [the coalescing atlas](HILBERT16-COALESCING-CAPTURE.md)
and [the next-atlas dichotomy](HILBERT16-NEXT-ATLAS.md). It reuses the
existing physical sections and does not invent a new closing map.

The identities are exact. The staged continuation that consumes them is
a written argument of the same class as coalescing §5.3, except that
kill-line Stage B is now an Interval Picard enclosure
([HILBERT16-STAGE-B.md](HILBERT16-STAGE-B.md)). It does **not**
pass G1: C2 remainders, Stage A/C, chart O, and complete first-hit
remain open. The fold I-map of
[HILBERT16-FOLD-LEADING.md](HILBERT16-FOLD-LEADING.md) supplies the
leading `sep = 0` derivative; it is not that remainder.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. The dichotomy is a matching strategy, not the derivative

On `sep = exp(-1/epsilon^2)`, every tested blow-up scale `sigma` bounds at
most one of `sigma * kappa` and the W-ratio to a *frozen* height
`h_max = epsilon^N`, `N in {0, 3, 4}`. That comparison is correct, and it
rules out leaving the outgoing factor

    (h_max / h_e)^{C epsilon},    h_e = Theta(epsilon^3 sep^2)

as an untracked `1 + o(1)` remainder.

The coalescing first-derivative formula already supplies
`exp(Psi_pre) <= C * sep^2` and the rectangle attenuation
`exp(-c * chi)`. The physical outgoing sensitivity is therefore bounded
by a *product*

    sep^2 * (h_1 / (epsilon^3 * mu * sep^2))^{C epsilon}
        * (first-root continuation from the sep-independent height h_1).

Requiring the middle factor to be `1 + o(1)` forced the spurious cut
`epsilon |log sep| <= 1`. Tracking it is the identity

    sep^2 * (h / (epsilon^3 * mu * sep^2))^alpha
        = sep^{2 - 2 alpha} * (h / (epsilon^3 * mu))^alpha,

with `alpha = C epsilon`. For `0 < epsilon < 1/C` and `sep in (0, 1]` the
sep-power is at most `1`. On the kill sequence it is
`exp((2 - 2 C epsilon) log sep) = exp(-(2 - 2 C epsilon)/epsilon^2)`,
which tends to zero faster than any `exp(O(1/epsilon))` growth.

Lean: [Hilbert16EntryExit.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16EntryExit.lean).
Python: `omnibias.dynamics.entry_exit_leading`.

## 2. Slow-line leading map

On the slow line the logarithmic height increment is the integrand
`x / B_-(x)` with `B_-(x) = (x - r1)(x - r2)`. The exact partial fraction

    x / ((x - r1)(x - r2))
        = (r1 / (r1 - r2)) / (x - r1) + (r2 / (r2 - r1)) / (x - r2)

and the double-root limit

    x / (x - rstar)^2 = 1 / (x - rstar) + rstar / (x - rstar)^2

are the leading entry-exit maps. They are not remainders. The W-ratio
explosion of a frozen section *is* this leading logarithm, not an error
term.

## 3. Three stages (written first-derivative cover of `sep > 0`)

Keep the incoming connector `|t_i| <= tbox < u^2/2` and
`lambda1 <= -lmin < 0`, `L >= Lmin`, `0 < sep <= sep0`.

**Stage A (χ-rectangle).** Coalescing §5.1–5.3: exit at
`y_0 = mu * sep^2`, `x_e = r1 + O(sep)`, and

    0 < dx_e / d kappa <= C * sep^2 * exp(-c * chi).

**Stage B (height inflation).** With `D = B + k y`, once `y` dominates
`|B| ~ sep^2` one has `dx/dy = epsilon k / x`, independent of height.
Integrating from `y_0` to a *sep-independent* `y_1 = O(1)` moves `x` by
`O(epsilon)`, so the orbit remains in a compact neighbourhood of
`rstar = -lambda1/2 >= lmin/2`. The variational factor of this stage is
exactly the tracked product of §1 with `h_1 = epsilon^3 y_1`.

**Stage C (first-root outgoing).** From the sep-independent height `h_1`
the outgoing continuation of [the first-root note](HILBERT16-ROOT-SADDLE.md)
§5 applies on a compact negative-`V` rectangle with uniform
`a_min ~ rstar/2`. Its integrating factor is `O(1)` in `kappa` and in
`sep`.

The product of A–C yields, for all sufficiently small `epsilon` and every
`sep in (0, sep0]`,

    0 < t_{o,kappa} <= C * epsilon^2 * exp(-c * chi).

The incoming lower bound `t_{i,kappa} >= c_3 * epsilon^2` does not see
`sep`. Hence

    0 < D_epsilon'(t_i) <= C * exp(-c * chi)

on the whole coalescing first-root compact, including the kill sequence.
This is a written first-derivative statement. It is not a C2 remainder
and not a first-hit completeness theorem.

## 4. What remains for G1

- `sep = 0`: the χ-rectangle has width zero; Stage A is empty. The
  double-root I-map of [HILBERT16-FOLD-LEADING.md](HILBERT16-FOLD-LEADING.md)
  is the leading map of chart C, not a remainder versus `B_eps`.
- Chart O: `L -> 0` sends `r1 -> 0`, so `a_min` fails at the colliding
  root. [Post-corridor matching](HILBERT16-POST-CORRIDOR.md) restores
  `T_e = Theta(eps^2)` at compact `x_*`; the
  [alpha-0 envelope](HILBERT16-HEIGHT-ENVELOPE.md) conserves `T-h` on
  the comparison ODE; height-section first-hit stays open.
- C2: the joined Rolle chain uses two `kappa` derivatives of `log D'`.
  Stage A–C control the first derivative only. Kill-line Stage B is
  [HILBERT16-STAGE-B.md](HILBERT16-STAGE-B.md). Kill-line Stage A
  wall identities are [HILBERT16-STAGE-A.md](HILBERT16-STAGE-A.md).
  The `chi_b` threshold is
  [HILBERT16-CHI-B.md](HILBERT16-CHI-B.md). Leading `dx_e` factors are
  [HILBERT16-DX-E-LEADING.md](HILBERT16-DX-E-LEADING.md). The
  uniform-in-`chi` bound is
  [HILBERT16-DX-E-UNIF.md](HILBERT16-DX-E-UNIF.md). The Stage-C
  `a_min` floor is [HILBERT16-STAGE-C.md](HILBERT16-STAGE-C.md).
  The Stage-C exit energy is
  [HILBERT16-STAGE-C-EXIT.md](HILBERT16-STAGE-C-EXIT.md). The Stage-C
  leading `T_h` floor is
  [HILBERT16-STAGE-C-TH.md](HILBERT16-STAGE-C-TH.md).
- Complete physical first-hit: the first-root note already refuses to
  assert that every height is admitted.
- Overlap `D ∩ C` at `sep = 0` is unmatched.

G1 does not pass. G4 is not opened. The parent flags stay false.

## 5. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_entry_exit_leading.py -q
uv run --no-sync python -m benchmarks.hilbert16_entry_exit_leading
lake build OmnibiasAnalytic.Dynamics.Hilbert16EntryExit
```
