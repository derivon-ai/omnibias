# Hilbert XVI: fold I-map leading derivative and C2

This companion continues [the saddle-node note](HILBERT16-SADDLE-NODE.md)
at `sep = 0`. The blow-up Gronwall factor `exp(C sigma kappa)` is a
majorant of a *frozen* scale, not the leading derivative of the slow-line
map. The exact double-root antiderivative supplies that derivative, and
its second `kappa` derivative of `log(dx/dkappa)` is the leading C2 of
chart C.

The identities are exact. They do **not** pass G1: the remainder versus
the actual field, chart O, complete first-hit, and a sealed orbit
continuation remain open.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. Implicit I-map

On the slow line at vanishing separation,

    B_-(x) = (x - r)^2,    dx / d tau = epsilon B_-(x),

so the height logarithm is the antiderivative

    I(x) = log|x - r| - r / (x - r).

A sep-independent exit height inverts `I(r - delta) = kappa` for a gap
`delta = r - x > 0`. Implicit differentiation is algebraic:

    dx / d kappa = delta^2 / (r - delta),
    d^2 x / d kappa^2 = - delta^3 (2 r - delta) / (r - delta)^3.

In particular `dx/dkappa -> 0` as `delta -> 0`, which is the large-`kappa`
tail. Gronwall `exp(C sigma kappa)` explodes on that tail; the I-map
does not.

## 2. Leading C2 of `log D'`

Along the same I-map,

    (log x')_kappa = - delta (2 r - delta) / (r - delta)^2,
    (log x')_kappa kappa = 2 r^2 delta^2 / (r - delta)^4.

The reciprocal gap `delta = r / kappa` (the large-`kappa` inversion of
`I(r - delta) ~ kappa`) yields

    dx / d kappa = r / (kappa^2 - kappa),
    (log x')_kappa kappa = 2 kappa^2 / (kappa - 1)^4 ~ 2 / kappa^2.

On `sep > 0` the χ-atlas leading logarithm is linear in `kappa`, so its
second `kappa`-difference vanishes identically. That is the leading C2
of charts D/F, not a remainder for the actual field.

## 3. Inner/outer remainder split

Write `B_eps = (x - r)^2 + epsilon * rho`. The relative remainder of the
slow-line integrand versus the I-map is

    (B_eps - B) / B = epsilon * rho / delta^2.

The inner/outer interface is `delta^2 = M * epsilon`. There the relative
remainder equals `rho / M`, independently of `epsilon`. Outer region
`delta^2 >= M * epsilon` therefore has relative remainder `O(1/M)` once
`rho` is bounded. The inner region `delta = O(sqrt(epsilon))` is the
existing fold blow-up chart; its remainder versus the actual field is
not sealed here.

The actual canonical remainder is not a free `rho`. With the documented
split `zeta(V, epsilon) = -1 + beta0 V + epsilon V Z`,

    B_eps - B_- = beta0 * epsilon * x^3 + epsilon^2 * x^3 Z,

so `rho = beta0 x^3 + epsilon x^3 Z` with `beta0 = 1/3`. On `x > 0` the
lift `beta0 epsilon x^3` is strictly positive. The inner leading field is
therefore the lifted fold

    B = (x - r)^2 + mu,    mu = beta0 epsilon r^3,

not `(x-r)^2`. In particular `B >= mu = Theta(epsilon)` near `x = r`,
`dx/dkappa = B/x` is `Theta(epsilon)` at the closest point, and the
leading inner C2 of `log(dx/dkappa)` is `O(epsilon)`. The Z remainder is
relatively `epsilon Z / beta0` on the lift: an inner-chart explosion of
`Z` does not occur. A Cauchy majorant for `Z` is sealed on the
`lambda0 = lambda1 = 0` slow-line embedding
([HILBERT16-CANONICAL-ZETA.md](HILBERT16-CANONICAL-ZETA.md)) and on a
declared fold compact of `(L, lambda1)`
([HILBERT16-FOLD-ZETA.md](HILBERT16-FOLD-ZETA.md)).

Lean: [Hilbert16FoldLeading.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16FoldLeading.lean).
Python: `omnibias.dynamics.fold_leading`.

## 4. What remains for G1

- Physical C2 of `log D'` off the lifted map (and on `sep > 0`):
  frozen-Z identities in
  [HILBERT16-PHYSICAL-C2.md](HILBERT16-PHYSICAL-C2.md); `Z_x` and a
  uniform-in-`eps` majorant remain.
- Chart O: outgoing first-hit of the large first-root **height** section on
  `L -> 0` (the x-corridor is
  [HILBERT16-OUTGOING-CORRIDOR.md](HILBERT16-OUTGOING-CORRIDOR.md); restored
  `T_e=Theta(eps^2)` hypotheses are
  [HILBERT16-POST-CORRIDOR.md](HILBERT16-POST-CORRIDOR.md); the `C=0`
  `T-h` envelope is
  [HILBERT16-HEIGHT-ENVELOPE.md](HILBERT16-HEIGHT-ENVELOPE.md); the `C=2`
  leading `|q|` ratio is
  [HILBERT16-Q-RATIO-C2.md](HILBERT16-Q-RATIO-C2.md); the `C=0` `k` jet
  is [HILBERT16-K-ZETA-REMAINDER.md](HILBERT16-K-ZETA-REMAINDER.md)).
- Complete physical first-hit: the first-root note already refuses to
  assert that every height is admitted.
- Sealed orbit-continuation remainder on the written Stage B/C cover of
  `sep > 0`.

G1 does not pass. G4 is not opened. The parent flags stay false.

## 5. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_fold_leading.py -q
uv run --no-sync python -m benchmarks.hilbert16_fold_leading
lake build OmnibiasAnalytic.Dynamics.Hilbert16FoldLeading
```
