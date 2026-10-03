# Saddle-node passage at vanishing separation

This companion continues [the χ-atlas](HILBERT16-COALESCING-CAPTURE.md) at
`sep = 0` and through the super-small sequence that killed chart D. It
reuses the incoming connector `|t_i| <= tbox < u^2/2` and the large
outgoing section of [the first-root note](HILBERT16-ROOT-SADDLE.md). It
does not invent a new closing map and does not apply Huzak–Kristiansen
Theorem 2.4 as a C2 or cyclicity statement.

Lean checks only exact double-root algebra and the linear fact that
`sep = 0` gives no χ-attenuation. This note does **not** claim a C2
remainder, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. Exact double-root field

Keep `Lmin <= L <= Lmax` and `lambda1 <= -lmin < 0`. Write
`sep = sqrt(lambda1^2 - 4 L)` when this is real and nonnegative, and
`rstar = -lambda1 / 2 >= lmin / 2`. At vanishing discriminant,
`sep = 0` and `L = rstar^2`, so the limiting outgoing quadratic is

    B_-(x) = L + lambda1 * x + x^2 = (x - rstar)^2.

The identity is polynomial. The actual `B_eps` differs by the established
`C^2` perturbation on each fixed radial interval. The slow-line
equilibrium is a saddle-node: `B_-'(rstar) = 0`, and the transverse
eigenvalue on `h = 0` remains `lambda_u = epsilon * r_eps` with
`r_eps = rstar + O(epsilon) > 0`.

The linear model of the χ-atlas with `sep = 0` gives `X_exit = 1`. There
is no exponential χ-attenuation. Any remainder at coalescence must come
from the nonlinear passage around the fold, not from a frozen `gamma`.

The compact-positive exclusion of
[the boundary-reduction note](HILBERT16-BOUNDARY-REDUCTIONS.md) is kept.
The central-strip coordinates `u = V / epsilon` and `W = h^epsilon / epsilon`
are the same. This note treats the complementary small-label itinerary.

## 2. Blow-up around the fold

In the outgoing chart `V = -epsilon * x`, `h = epsilon^3 * y`,
`tau = epsilon * (normal time)`,

    x_tau = epsilon * (B_eps(x) + k y),    y_tau = x * y,

with `B_eps(x) = (x - rstar)^2 - (sep / 2)^2 + O(epsilon)` on compact
`x` intervals. Fix a scale `sigma > 0` and set

    xi = (x - rstar) / sigma,    eta = y / sigma^2.

Exactly,

    B_-(x) = sigma^2 * xi^2 - (sep / 2)^2,
    xi_tau = (epsilon / sigma) * (sigma^2 * xi^2 - sep^2 / 4 + k * sigma^2 * eta
                                  + O(epsilon)),
    eta_tau = (rstar + sigma * xi) * eta.

A compact `(xi, eta)` box is available only after `sigma` is chosen so
that the displayed coefficients stay bounded. Two natural choices are
recorded.

**Fold scale.** `sigma = sqrt(epsilon)`. Then `eta`-exit can be fixed at
an `O(1)` value `eta0`, so the physical exit height is
`h_e = epsilon^3 * y = epsilon^3 * sigma^2 * eta0 = O(epsilon^4)`,
independent of `sep`. The outgoing variational factor of the first-root
note satisfies

    (h_max / h_e)^{C epsilon} = exp(O(epsilon log(1 / epsilon))) -> 1

uniformly in `sep`, including `sep = 0`. This removes the
`epsilon |log sep|` restriction that chart D needed for a shrinking
`y0 = mu * sep^2`.

**Separation scale.** `sigma = sep` when `sep > 0`. Then `chi = O(1)`
corresponds to `kappa = O(1 / sep)`, and the variational integral below
stays `O(1)` on that locus. The exit height is again
`h_e = O(epsilon^3 sep^2)`, and the outgoing factor is the chart-D term
`exp(O(epsilon |log sep|))`, which is unbounded on the kill sequence of
the χ-atlas.

## 3. First-derivative tension

The planar event identity of the first-root note is unchanged. On a
compact blow-up box, `B_eps' = 2 (x - rstar) + O(epsilon) = 2 sigma * xi
+ O(epsilon)`, so

    integral epsilon (B_eps' + y k_x) d tau = O(epsilon sigma) * T,

with residence `T` at least `log(eta0 / eta_e) / X` and
`X = O(1)`. The entry height satisfies `eta_e = y_e / sigma^2` and
`y_e <= exp((S_pre - kappa) / epsilon)`. Thus

    exp integral epsilon B_eps' d tau
        <= exp(C * epsilon * sigma * T)
        <= C' * exp(C * sigma * kappa) * (poly(sigma, epsilon)).

The product `sigma * kappa` is the obstruction of that Gronwall
majorant. It is not the leading derivative of the slow-line map: the
[fold I-map](HILBERT16-FOLD-LEADING.md) gives
`dx/dkappa = delta^2 / (r - delta)`, which decays as `~ r / kappa^2`.

On a `chi = O(1)` locus, `kappa = chi * r1 / sep` with `r1` bounded
below on `L >= Lmin`. Then

    sigma * kappa = (sigma / sep) * chi * r1.

- If `sigma <= C * sep`, then `sigma * kappa` stays bounded on
  `chi = O(1)`, but `h_e = O(epsilon^3 sigma^2)` is `O(epsilon^3 sep^2)`
  and the outgoing factor of chart D is back.
- If `sigma >= sqrt(epsilon)`, then the outgoing factor is uniform, but
  on the χ-atlas kill sequence

      sep_n = exp(-1 / epsilon_n^2),    kappa_n = 1 / sep_n,
      lambda1 = -3,    L_n = (9 - sep_n^2) / 4,

  one has `sigma_n / sep_n >= sqrt(epsilon_n) * exp(1 / epsilon_n^2)`,
  so `sigma * kappa` is unbounded. The first-derivative event factor
  `exp(C sigma kappa)` is then unbounded.

No single positive scale `sigma(epsilon, sep)` makes both
`sigma * kappa` and `(h_max / h_e)^{C epsilon}` bounded on that
sequence. This is an obstruction to this blow-up pattern, not a
counterexample to finite cyclicity.

**Saddle-node first-derivative verdict.** The fold scale absorbs
`sep -> 0` only on the restricted tail `kappa = O(1 / sigma) =
O(epsilon^{-1/2})`. It does not absorb the `chi = O(1)` kill sequence.
The separation scale reproduces chart D and does not absorb that
sequence either. A uniform first derivative at `sep = 0` and on
`0 < sep < exp(-1 / sqrt(epsilon))` for all admitted `kappa` is not
obtained.

## 4. C2 remainder

The joined / varying-detuning chain requires two `kappa` derivatives of
`log D'` at fixed physical coefficients, including the curved-scale
acceleration of
[the singular-kappa note](HILBERT16-SINGULAR-KAPPA-JETS.md). Chart D
never supplied those derivatives. Differentiating the event identity
once more produces a factor of the residence time, or of
`partial_kappa log eta_e ~ -1 / epsilon`, contracted against
`epsilon sigma`. The same product `sigma * kappa` appears, now in the
second derivative.

Because the first derivative is already unbounded on the kill sequence
for every scale in section 3, no C2 remainder is claimed. Huzak–Kristiansen
entry-exit, used as a value formula on compact interior sectors in
[singular transport](HILBERT16-SINGULAR-TRANSPORT.md), is not differentiated
here and is not a C2 theorem at `sep = 0`.

**Saddle-node C2 verdict: fail.** The named falsifier is the same
sequence as coalescing §5.5, now seen as a scale tension rather than a
missing chart label.

## 5. What is proved

- The exact double-root identity `B_- = (x - rstar)^2`.
- The linear identification `X_exit = 1` at `sep = 0`.
- That a fold-scale exit height independent of `sep` makes the first-root
  outgoing factor `(h_max / h_e)^{C epsilon}` uniform in `sep`.
- That this uniformity is incompatible, on the kill sequence, with a
  bounded event factor `exp(C sigma kappa)` at `chi = O(1)`.

No physical C2 remainder, no cycle bound, and no G1 pass follow.
The scale tension is at `L >= Lmin`. It does not block the separate
`L -> 0` layer recorded in [the shrinking-root note](HILBERT16-SHRINKING-ROOT.md).
The follow-up [two-blow-up covering](HILBERT16-TWO-BLOWUP.md) rewrites the
outgoing factor as a W-ratio and still fails on the same kill sequence.
[The next-atlas note](HILBERT16-NEXT-ATLAS.md) writes that tension as a
scale dichotomy and records that the kill sequence remains admitted.

## 6. Reproduction

```bash
uv run --no-sync python benchmarks/hilbert16_saddle_node.py \
  --output artifacts/hilbert16/saddle_node.json
lake build OmnibiasAnalytic.Dynamics.Hilbert16SaddleNode
```
