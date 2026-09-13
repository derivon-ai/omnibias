# Coalescing-root χ-atlas, double-root estimates, and G1/G4 verdict

This companion records the next Hilbert XVI research increment on the
selected `I_2^1` / `I_4^1` small-label itinerary. It reuses the existing
physical sections, positive-label overlaps, and connected admission of
the [first-root saddle](HILBERT16-ROOT-SADDLE.md),
[joined positive-base](HILBERT16-JOINED-POSITIVE-BASE.md),
[varying-detuning](HILBERT16-VARYING-DETUNING.md), and
[boundary-reduction](HILBERT16-BOUNDARY-REDUCTIONS.md) notes. It does
not invent a new closing map.

Lean checks only the exact linear χ-identities and the frozen-exponent
obstruction. Finite replay does not replace a continuum remainder. This
note does **not** claim G1, local graphic cyclicity, or Hilbert XVI.

Checked against the program ledger on **2026-09-13**. No novelty or
external acceptance is asserted.

## 0. Honesty and kill criterion

A sequence of admitted passages falsifies the atlas if it escapes the
charts below, loses a required denominator or section margin, or makes a
normalized remainder used by a downstream zero-count unbounded. Primitive
bounds from `omnibias.core.verified.asymptotic_jet` alone do not pass G1.

The parent flags stay false throughout this note:

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

Sections 8–10 record why G1 and G4 fail, with named escaping sequences.
Algebraic G5 remains off the critical path.

## 1. Linear model, χ, and the frozen-exponent obstruction

The missing uniformity is already exact on

    Xdot = -epsilon * sep * X,    Ydot = r * Y,
    X(0) = 1,    Y(0) = exp(-kappa / epsilon),    exit Y = 1,

with `epsilon, sep, r, kappa > 0`. The flight time is `kappa / (epsilon * r)`,
so

    X_exit = exp(-(sep / r) * kappa).

Define the matching coordinate

    chi = (sep / r) * kappa.

Then `X_exit = exp(-chi)` and `d X_exit / d chi = -exp(-chi)`, uniformly
in `sep`. The kappa sensitivity is

    |d X_exit / d kappa| = (sep / r) * exp(-chi).

There do not exist `C, gamma > 0`, independent of all sufficiently small
`sep > 0`, such that this is at most `C * exp(-gamma * kappa)` for every
sufficiently large `kappa`. Indeed, for any such `C, gamma` and any
`s0 > 0`, the choice `sep = min(s0, r * gamma / 2)` makes the ratio

    |d X_exit / d kappa| / (C * exp(-gamma * kappa))
        = (sep / (r * C)) * exp((gamma - sep / r) * kappa)

grow without bound as `kappa -> infinity`. This is the obstruction of
[the cyclicity calculus](HILBERT16-CYCLICITY-CALCULUS.md), section 4. It
obstructs a frozen first-root exponent, not finite cyclicity.

The two limits `sep -> 0` at fixed `chi` and `kappa -> infinity` at fixed
`sep / r` are different. The first sends `kappa = chi * r / sep` to
infinity while keeping the linear exit of order one. The second is the
strict first-root tail already treated in
[HILBERT16-ROOT-SADDLE.md](HILBERT16-ROOT-SADDLE.md).

[Hilbert16ChiScale.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ChiScale.lean)
proves the exact exit identification, the `chi` and `kappa` derivatives,
the sensitivity ratio, the reciprocal-separation threshold identity
`chi(sep, r, K / sep) = K / r`, and the frozen-exponent obstruction.
Those statements do not mention the quadratic field.

## 2. Actual first-root scale (notation)

On the canonical normal field of the first-root note,

    Vdot = f(V) + h * g(V, h),    hdot = -V * h,
    f(V) = -L * epsilon^3 + lambda1 * epsilon^2 * V
           + epsilon * V^2 * zeta(V, epsilon),

write `delta = lambda1^2 - 4 L` and `sep = sqrt(delta)` when
`delta >= 0`. The outgoing chart is `V = -epsilon * x` with

    B_eps(x) = L + lambda1 * x - x^2 * zeta(-epsilon * x, epsilon),
    B_-(x) = L + lambda1 * x + x^2 = (x - r1) * (x - r2),

    r1 = (-lambda1 - sep) / 2,    r2 = (-lambda1 + sep) / 2,
    rstar = -lambda1 / 2.

The exact saddle at `(V, h) = (-epsilon * r_eps, 0)` has

    eta_eps = |lambda_s| / lambda_u
            = epsilon * (sep / r1 + O(epsilon)).

A transverse entry of size `exp(-kappa / epsilon)` is therefore attenuated
on the scale `exp(-(sep / r1) * kappa)`. The actual matching coordinate
on this branch is

    chi = (sep / r1) * kappa,

not a new normal-form coordinate. The first-root theorem holds on a
strict compact `sep >= sep0 > 0` and `L >= Lmin > 0`. It excludes
`sep -> 0` and `L -> 0`.

Do not confuse `sep` with the boundary-reduction parameter
`s_BR = lambda1 / 2` of [HILBERT16-BOUNDARY-REDUCTIONS.md](HILBERT16-BOUNDARY-REDUCTIONS.md).
Coalescence `sep -> 0` with `lambda1 <= -lmin < 0` is their
`delta_BR -> 0` at bounded negative `s_BR`.

## 3. Finite weighted charts

The G1 request is a finite weighted chart description spanning the
no-root, double-root, and first-root limits, together with the
`L -> 0` layer. The charts below reuse existing sections.

| Chart | Parameter set | Existing control | Open in this note |
| --- | --- | --- | --- |
| N (no-root) | `Delta = 4 L - lambda1^2 >= chi_N > 0`, `L` compact, `lambda1 <= -lmin` | Joint matching, two-scale, varying detuning: at most three admitted cycles | Degenerating compact margins; not every nearby cycle |
| F (first-root) | `sep >= sep0 > 0`, `L >= Lmin > 0`, `lambda1 <= -lmin` | Root-saddle: at most two admitted cycles | Admission of every height or nearby cycle |
| D (double-root / shrinking rectangle) | `0 < sep <= sep0`, `L >= Lmin`, `lambda1 <= -lmin` | Linear χ-model; section 5 below | Super-small `sep`, including `sep = 0`; C2 remainders |
| C (central fold strip) | Boundary-reduction chart `u = V / epsilon`, `W = h^epsilon / epsilon` | Compact positive endpoints excluded near a nonzero `s_BR` | Shrinking labels at `sep = 0`; `L -> 0` corner |
| O (origin / shrinking first root) | `L -> 0` | Named as excluded by the exponential and first-root notes | Uniform `a_min`; first-hit completeness |

Compact positive-endpoint passages near a discriminant point are already
excluded. Chart D is the complementary shrinking-label coalescence
regime. Chart O is a different degeneration: `r1 -> 0`.

Kill sequences for this atlas include any admitted orbit that leaves the
selected small-label incoming interval `|t_i| <= tbox < u^2 / 2` with
the fixed `u in (0, 2)` of the first-root note, or that loses the large
outgoing transversality margin.

## 4. Matching rules

On overlaps the coordinates are the same physical labels.

- F ∩ D: `sep` near `sep0`. Chart F uses a frozen `gamma > 0`. Chart D
  uses `chi`. They agree because `chi = (sep / r1) * kappa` and `sep / r1`
  has a positive lower bound on F, so a large-`chi` tail is a large-`kappa`
  tail.
- D ∩ C: the fold strip of the boundary-reduction note, with shrinking
  rather than compact-positive labels. The overlap is only claimed for
  `epsilon |log sep|` bounded (section 5). It is not claimed for
  `sep = 0` or `sep < exp(-1 / sqrt(epsilon))`.
- N ∩ F: opposite discriminant signs, already left to a separate
  coalescence analysis by the first-root and joint-matching notes.
- O ∩ anything: not matched. The first-root rectangle requires a uniform
  `a_min > 0`.

The identity `chi(sep, r1, K / sep) = K / r1` is the exact reason a
pre-rectangle time `S_pre = O(1 / sep)` produces a `chi`-threshold of
order one, independent of `sep`. That identity does not restore a frozen
`gamma`.

## 5. Actual shrinking-rectangle passage (chart D)

Fix the first-root coefficient compact except the lower bound on `sep`:

    Lmin <= L <= Lmax,    lambda1 <= -lmin < 0,
    0 < sep <= sep0,

with `sep0` small enough that `r1 >= rstar / 2 >= lmin / 4` and
`r2 - r1 = sep`. Keep the incoming connector at a fixed `u in (0, 2)`
and the large outgoing section of the first-root note.

### 5.1 Shrinking trapping rectangle

Fix `theta in (0, 1/4)`. After the implicit-function displacement
`r_eps = r1 + O(epsilon)`, set

    a = r_eps - theta * sep,    bnd = r_eps + theta * sep.

For small `epsilon` and `sep`, `a >= a_min := lmin / 8 > 0`. On the
limiting quadratic,

    B_-'(x) = 2 (x - rstar) <= -sep (1 - 2 theta) < 0

throughout `[a, bnd]`. After the established `C^2` perturbation of
`B_eps` and `k_x = O(epsilon^2)`, there is a uniform factor `1/2` such
that

    B_eps'(x) + y * k_x <= -sep (1 - 2 theta) / 2

on the rectangle once `0 <= y <= y0`. The wall values are

    B_-(a) = theta (1 + theta) sep^2.

Choose `y0 = mu * sep^2` with `mu` small enough that `kmax * y0` is at
most half of `B_eps(a)`. A fixed `y0` independent of `sep` is
impossible: it would violate the inward-wall condition as `sep -> 0`.
The first-root proof used a fixed rectangle width and a fixed `y0`,
which is why it cannot be continued to coalescence.

The left wall still points right and the right wall left. The first-root
sign argument that `D` cannot first hit zero from above remains: at a
hypothetical `D = 0`, `D_tau = D_y * x * y > 0`. Thus `x` increases
through the rectangle.

### 5.2 Pre-rectangle time and the χ-threshold

On the slow line, `S_pre = integral_0^a x / B_eps(x) dx`. The limiting
integrand is `x / ((x - r1) (x - r2))`, with antiderivative

    (1 / (r1 - r2)) (r1 log |x - r1| - r2 log |x - r2|).

The potentially divergent `log sep` pieces cancel at leading order. The
remainder is `O(1 / sep)`, with a coefficient bounded on the compact
`L, lambda1` set. After the `O(epsilon)` perturbation there is `K`,
depending only on that compact and on `theta`, such that
`S_pre <= K / sep`.

The exact radial height before the rectangle gives
`y_e <= exp((S_pre - kappa) / epsilon)`. The threshold

    kappa >= kappa_b := (K + 1) / sep

places `y_e` below `y0`. The corresponding `chi` threshold is

    chi_b = (sep / r1) * kappa_b <= (8 / lmin) (K + 1),

independent of `sep`. This is the content of the reciprocal-separation
identity, not a frozen `gamma`.

### 5.3 Event derivative in χ

The planar-divergence identity of the first-root note, section 4, is
unchanged. It yields

    dx_e / d kappa = (A_eps / x_e) * exp(Psi_pre)
                     * exp integral epsilon (B_eps' + y * k_x) d tau,

with `A_eps = L + k(0, H) * exp(-kappa / epsilon) >= Lmin / 2` for small
`epsilon`. Residence in the rectangle is at least
`log(y0 / y_e) / X` with `X <= 2 rstar`. The integrand `B_eps' + y k_x`
is at most `-sep (1 - 2 theta) / 2`, so the exponential integral is at
most `C * exp(-c * chi)` with

    c = (lmin / 8) * (1 - 2 theta) / (2 * sqrt(Lmax) + 1) > 0.

On the slow line, `Psi_pre = log B_eps(a) - log B_eps(0)` plus the
exact `k` remainder. Thus `exp(Psi_pre) = O(sep^2)` there. With
`y <= y0 = mu * sep^2` the denominator `B + k y` stays a definite
fraction of `B` on the pre-rectangle after reducing `mu`, because
`B` is bounded below by a positive constant on any compact subinterval
of `[0, r1 - 2 theta * sep]` and is comparable to `sep^2` only on an
interval of length `O(sep)` where `y` has not yet grown past `y0`.
Therefore `exp(Psi_pre) <= C * sep^2` for the actual incoming height,
and

    0 < dx_e / d kappa <= C * sep^2 * exp(-c * chi).

Outgoing continuation to the selected large section follows the
first-root bootstrap with the smaller exit height
`h_e = epsilon^3 * y0 = O(epsilon^3 sep^2)`. The variational factor

    (h_max / h_e)^{C epsilon}
        = exp(O(epsilon log(1 / epsilon) + epsilon |log sep|))

remains `1 + o(1)` whenever `epsilon |log sep|` is bounded, in
particular on `sep >= exp(-1 / sqrt(epsilon))`. The physical outgoing
label then satisfies

    0 < t_o,kappa <= C * epsilon^2 * sep^2 * exp(-c * chi)

on that restricted set. The incoming lower sensitivity
`t_i,kappa >= c3 * epsilon^2` of the first-root note, section 6, uses
only the fixed connector at `V = u` and does not see `sep` in its
leading lower bound. The singular slope therefore obeys

    0 < D_eps'(t_i) <= C * sep^2 * exp(-c * chi)

uniformly for all `L, lambda1` in the compact, all
`sep in (0, sep0]`, and all sufficiently small `epsilon` satisfying
`epsilon |log sep| <= 1`.

### 5.4 Overlap with chart F and the compact-kappa band

On `sep >= sep0 / 2` the new bound implies a frozen positive `gamma`,
recovering the first-root tail after shrinking `c` by the compact lower
bound of `sep / r1`. The existing finite exponential band of the
first-root note still covers the compact-`kappa` overlap with the
grazing / thin-height region. No separate cycle counts are added.

### 5.5 What this does not prove

The estimate requires `epsilon |log sep| <= 1`. It does not cover

- `sep = 0` (the rectangle has width zero; the saddle is a saddle-node),
- `0 < sep < exp(-1 / sqrt(epsilon))` (the outgoing variational factor
  `(h_max / h_e)^{C epsilon}` is not proved bounded),
- C2 remainders in `kappa` or `chi` on chart D (only the first
  derivative of the incoming/outgoing labels),
- admission of every positive height, or of every nearby cycle.

The sequence

    sep_n = exp(-1 / epsilon_n^2),    epsilon_n = 1 / n,
    L = 1,    lambda1 = -3,    kappa_n = 1 / sep_n

has `chi_n` of order one and `epsilon_n |log sep_n| = 1 / epsilon_n`
unbounded. It is a kill sequence for the outgoing remainder of
section 5.3, not a counterexample to finite cyclicity. Any G1 claim
that ignores it is false.

## 6. Central fold at vanishing separation

The boundary-reduction central chart already excludes compact positive
input/output labels when `|s_BR| >= s0 > 0` and their `delta_BR` is
small. That exclusion is kept. It does not supply a C2 small-label
passage at `sep = 0`.

At exact coalescence, `B_-(x) = (x - rstar)^2` with `rstar >= lmin / 2`.
The linear model has `sep = 0` and `X_exit = 1`: there is no
exponential `chi`-attenuation. A saddle-node / entry-exit description
is a different theorem. The published Huzak–Kristiansen entry-exit
hypotheses require a strict nonvanishing weighted drift and do not, by
themselves, give the derivatives used by the joined Rolle chain. This
note does not apply that theorem at `sep = 0`.

## 7. Origin layer `L -> 0` (chart O)

The exponential and first-root notes both exclude `L -> 0`. Keep
`lambda1 <= -lmin < 0` first. Then

    sep = sqrt(lambda1^2 - 4 L) -> |lambda1|,
    r1 = (-lambda1 - sep) / 2 = 2 L / (|lambda1| + sep) -> 0.

The outgoing saddle approaches `V = 0`. The uniform `a_min > 0` of
charts F and D fails. The incoming connector at a fixed `V = u > 0`
remains, but outgoing continuation from a rectangle that shrinks toward
the centre is not the first-root argument.

The sequence

    L_n = 1 / n,    lambda1 = -2,    sep_n -> 2,    r1_n -> 0

escapes every chart that requires a uniform positive radial wall. It is
admitted by the first-root coefficient inequalities except `L >= Lmin`.

If also `lambda1 -> 0`, one is in the boundary-reduction corner
`lambda1 = kappa_BR * l1`, `lambda0 = kappa_BR^2 * l0`. That chart still
requires its own height normalization; it is not an application of the
interior or χ-rectangle theorems. First-hit domains are not proved
there.

No first-hit completeness statement is claimed on chart O. In particular
this note does not assert that every small centre height reaches the
selected small-label section when `r1 -> 0`.

## 8. G1 verdict: fail

G1 requires all four of:

1. a finite weighted chart description spanning no-root, double-root, and
   first-root limits, including the `L -> 0` layer;
2. complete physical first-hit domains for the selected connector;
3. uniform variational remainders in every derivative later used for a
   zero count;
4. endpoint matching on overlaps.

Item 1 is only a labelled atlas. Charts N and F are existing written
arguments. Chart D has a first-derivative χ-bound under
`epsilon |log sep| <= 1`. Charts C (at `sep = 0`) and O are not closed.

Item 2 fails: the first-root note already refuses to assert that every
height is admitted, and chart O has no first-hit theorem.

Item 3 fails: the joined / varying-detuning zero count uses two `kappa`
derivatives of `log D'`. Chart D supplies no C2 remainder. The
super-small-`sep` sequence of section 5.5 makes even the first-derivative
outgoing factor unproved.

Item 4 fails on O ∩ D and on D ∩ C at `sep = 0`.

**G1 does not pass.** The named falsifiers are the super-small
separation sequence of section 5.5 and the shrinking-root sequence of
section 7. Do not proceed to a cyclicity claim on a partial atlas.

## 9. G4 verdict: fail

G4 requires that every degenerating sequence of nearby cycles has a
subsequence in a chart carrying an open original-parameter neighbourhood
with a uniform count, including identity fibres and incident sides.

G1 did not pass, so G4 is not available. Independently, the first-root,
joined, and varying-detuning theorems all state that every nearby cycle
is not asserted to enter the selected small-label tube. Identity fibres
are isolated by the polynomial cyclicity checker only for exact
polynomial models, not for the actual singular return.

**G4 does not pass.** The coverage hole is: nearby cycles that miss the
selected tube, identity fibres of the actual return, and every sequence
that follows the G1 falsifiers.

No local-cyclicity sentence for `I_2^1` / `I_4^1` is licensed.

## 10. Reassessment of the parent

The parent remains open. The high-water mark is unchanged except for a
sharper obstruction and a restricted χ-estimate:

- existing N and F cycle bounds stand;
- a written first-derivative χ-bound holds on chart D under
  `epsilon |log sep| <= 1`;
- G1 and G4 fail by the named sequences above;
- `full_hilbert16_solved` stays false.

The saddle-node, shrinking-root, and two-blow-up increments were written
next, in that order, on the same sections. All fail:

1. [Saddle-node](HILBERT16-SADDLE-NODE.md) records a fold-versus-separation
   scale tension. No single `sigma(epsilon, sep)` bounds both
   `sigma * kappa` and `(h_max / h_e)^{C epsilon}` on the super-small
   sequence of section 5.5. The first derivative at `sep = 0` and on
   `0 < sep < exp(-1 / sqrt(epsilon))` for all admitted `kappa` is not
   obtained, so no C2 remainder is claimed.
2. [Shrinking-root](HILBERT16-SHRINKING-ROOT.md) retains incoming first-hit
   on the selected `V = u` connector, but the outgoing saddle
   `V = -epsilon * r1` collides with the centre along `L_n = 1 / n`,
   `lambda1 = -2`. Continuation to the large outgoing section loses its
   uniform margin. SR2 rematch onto the existing height / grazing /
   exponential coefficient sets fails on the selected small-label
   itinerary.
3. [Two-blow-up](HILBERT16-TWO-BLOWUP.md) rewrites the outgoing factor as
   `(W_max / W_e)^C`. Chart WF makes that ratio uniform in `sep`; chart WS
   makes `sigma * kappa` of order one on `chi = O(1)`. No pair covers the
   super-small sequence. No C2 remainder is claimed.
4. [Next atlas](HILBERT16-NEXT-ATLAS.md) derives the leading blow-up
   `kappa`-jet, tests logarithmic charts LI/WL, and records a scale
   dichotomy: every positive `sigma` bounds at most one of `sigma * kappa`
   and the W-ratio to an existing `epsilon^N` section. The super-small
   sequence remains admitted. The two kill axes are disjoint at fixed
   `lambda1 < 0`. No C2 remainder is claimed.

**G1 remains failed.** The four G1 items of section 8 are still open.
G4 is not opened. G2, remaining DRR cases, and algebraic G5 stay
separate. The [chart-cell ledger](HILBERT16-CHART-CELLS.md) records the
named cells; a complete list of labels is not G1.

A finite replay, a fitted jet, or a Lean identity does not close those
obligations.

## 11. Formal and executable scope

[Hilbert16ChiScale.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ChiScale.lean)
contains the linear χ-identities and the frozen-exponent obstruction.
[Hilbert16SaddleNode.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16SaddleNode.lean)
contains the exact double-root and shrinking-root identities.
[Hilbert16TwoBlowup.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16TwoBlowup.lean)
contains the exact W-coordinate identities.
[Hilbert16ScaleDichotomy.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ScaleDichotomy.lean)
contains the exact blow-up, event-jet, joint-axis, and log-inner
identities. None of them mentions physical C2 remainders or cycle counts.

The [coalescing-capture benchmark](../../benchmarks/hilbert16_coalescing_capture.py)
replays the exact linear identities, the first-root eigenvalue ratio,
the wall value `B_-(a) = theta (1 + theta) sep^2`, the slow-line
`Psi_pre` logarithm, the reciprocal-separation threshold, and a concrete
frozen-`gamma` failure. The
[saddle-node benchmark](../../benchmarks/hilbert16_saddle_node.py)
replays the double-root algebra and a concrete scale-tension witness.
The [two-blow-up benchmark](../../benchmarks/hilbert16_two_blowup.py)
replays the W-identities, named scale paths, one Maletto type, and the
chart-cell ledger. The
[next-atlas benchmark](../../benchmarks/hilbert16_next_atlas.py)
replays the scale-dichotomy identities and the shared findings. All
reject an unresolved residual. Their honesty
blocks keep every parent flag false.

```bash
uv run --no-sync python benchmarks/hilbert16_coalescing_capture.py \
  --output artifacts/hilbert16/coalescing_capture.json
uv run --no-sync python benchmarks/hilbert16_saddle_node.py \
  --output artifacts/hilbert16/saddle_node.json
uv run --no-sync python benchmarks/hilbert16_two_blowup.py \
  --output artifacts/hilbert16/two_blowup.json
uv run --no-sync python benchmarks/hilbert16_next_atlas.py \
  --output artifacts/hilbert16/next_atlas.json
uv run --no-sync python -m benchmarks.hilbert16_program --lean
lake build OmnibiasAnalytic.Dynamics.Hilbert16ChiScale
lake build OmnibiasAnalytic.Dynamics.Hilbert16SaddleNode
lake build OmnibiasAnalytic.Dynamics.Hilbert16TwoBlowup
lake build OmnibiasAnalytic.Dynamics.Hilbert16ScaleDichotomy
```

These checks are a reproducibility record of scoped identities, not a
global proof.
