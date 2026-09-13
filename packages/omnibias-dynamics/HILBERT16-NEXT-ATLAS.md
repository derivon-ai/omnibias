# Next-atlas findings: scale dichotomy and named exclusions

This companion records six on-path attempts against the two kill sequences
of [the two-blow-up note](HILBERT16-TWO-BLOWUP.md) and
[the shrinking-root note](HILBERT16-SHRINKING-ROOT.md). Findings are
shared: a fail in one chart is a constraint on the next. The increment
reuses the incoming connector `|t_i| <= tbox < u^2/2` and existing
outgoing first-root / height language. It does not invent a new closing
map and does not apply Huzak–Kristiansen Theorem 2.4 as a C2 or cyclicity
statement.

Discovery, named `lim` paths, and symbolic reduction sit upstream: they
propose the identities below. They do not replace a continuum remainder.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

On the first named sequence, `sep` is the unfolding distance to the
double root inside `L >= Lmin`, `lambda1 = -3`, not the frozen pair
`L = 1`, `lambda1 = -3` (that pair has algebraic separation `sqrt(5)`).

## Shared findings

| Idea | What it proved | What it falsified | Constraint on the others |
| --- | --- | --- | --- |
| A. HK two-step derived | From the actual blow-up field, the leading event exponent is `I = (sigma / X) kappa + (2 epsilon sigma / X) log sigma`. It is affine in `kappa`; the first difference is `sigma / X`; the second vanishes. | Quoting Theorem 2.4 as C2 or cyclicity on this tube | The inner `kappa`-jets are not the hole. Matching to an existing outgoing section still carries the W-ratio obstruction. |
| B. Log / intermediate chart | On the kill sequence, `tau = epsilon log(1/sep) = 1/epsilon`. Charts LI (`tau`) and WL (`Lambda = epsilon log(1/W)`) do not produce a compact third scale that bounds both sides. | A scale `sigma` that is both `>= sqrt(epsilon)` and `<= sep` when `sep << sqrt(epsilon)`, including logarithmic `sigma` | Every later chart may keep at most one of bounded `sigma * kappa` and a bounded W-ratio to an existing `epsilon^N` section. |
| C. Residence-time exclusion | Incoming first-hit on the selected connector is retained. `chi = 1 / rstar` is order one. Compact-positive residence exclusion does not apply. | Inadmissibility of `sep = exp(-1/epsilon^2)` on this tube | The sequence remains a G1 falsifier. Vanishing `chi`-chart drift excludes Theorem 2.4; it does not exclude the passage. |
| D. Outgoing-section redesign | Existing sections at `h ~ 1`, `h ~ epsilon^3`, and `h ~ epsilon^4` all have exploding W-ratio on the kill sequence when `sigma = sep`. | A smaller *existing* outgoing section with bounded W-ratio | Only `h ~ epsilon^3 sep^2` would bound the ratio. That is a new closing map, which is refused. |
| E. Joint `(sep, L)` | Exact identity `2 r1 + sep + lambda1 = 0`. At fixed `lambda1 < 0` the two kill axes are disjoint. | A joint chart covering both named sequences at `lambda1 <= -lmin` | The joint origin `sep -> 0` and `L -> 0` forces `lambda1 -> 0`, already named as the boundary-reduction corner. SR2 rematch stays failed. |
| F. Exact proposers | Five rational residuals (blow-up height, height ratio, leading event, joint-axis sum, second `kappa` difference) replay as `PROVED`. | Unused catalog hits as progress | FiniteFamily / verdict collapse certify residuals, not a C2 remainder. |

**Written falsifier.** On the `chi = O(1)` sequence
`sep = exp(-1/epsilon^2)`, `kappa = 1/sep`, every positive scale
`sigma(epsilon, sep)` bounds at most one of

    sigma * kappa = (sigma / sep) * chi * r1

and

    (h_max / h_e)^{C epsilon},    h_e = Theta(epsilon^3 sigma^2),

when `h_max` is an existing section `epsilon^N` with `N in {0, 3, 4}`.
If `sigma = O(sep)` then `epsilon |log sigma| = 1/epsilon + O(1)` and the
W-ratio explodes. If the W-ratio stays bounded then
`sigma >= exp(-K / epsilon)` for some `K`, so `sigma / sep` explodes.
No third scale exists between `sqrt(epsilon)` and `sep`. Chart LI has
`tau = 1/epsilon`, which leaves every compact. Chart WL makes `Lambda_e`
order one while the variational factor remains `exp(Delta Lambda / epsilon)`.

## 1. HK two-step derived, not quoted

In the existing blow-up coordinates of
[the saddle-node note](HILBERT16-SADDLE-NODE.md),

    xi = (x - rstar) / sigma,    eta = y / sigma^2,
    eta_tau = (rstar + sigma * xi) * eta.

Residence to a compact `eta = eta0` from `eta_e = y_e / sigma^2` satisfies
`integral X d tau = log(eta0 / eta_e)` with `X = rstar + sigma xi`.
The planar event integral of the first-root note is

    I = integral epsilon (B_eps' + y k_x) d tau,
    B_eps' = 2 sigma * xi + O(epsilon).

Hence `|I| <= C epsilon sigma T`. With
`y_e <= exp((S_pre - kappa) / epsilon)` this is

    I = (sigma / X_*) kappa + (2 epsilon sigma / X_*) log sigma
        + O(epsilon kappa),

the displayed leading term being the actual-field analogue of a first
blow-up followed by a log correction at the even-order fold. Differentiating
in `kappa` at fixed `(epsilon, sep, sigma)` gives `sigma / X_*`. The second
`kappa` difference of that leading term is identically zero.

Huzak–Kristiansen Theorem 2.4 needs compact positive labels and a
nonvanishing weighted drift. The selected tube has shrinking outgoing
labels. On the kill sequence the `chi`-chart drift is `sep -> 0`. Those
hypotheses fail. The theorem is a named exclusion, not a cyclicity quote.
Dulac smoothness in `(epsilon, epsilon log(1/epsilon))` does not contain
the locus `epsilon |log sep| = 1/epsilon`.

The inner `kappa`-jet is therefore not the obstruction. The hole remains
the match to an existing outgoing section, which idea B writes as a
dichotomy.

## 2. Logarithmic and intermediate charts

Chart WF uses `sigma = sqrt(epsilon)`. Chart WS uses `sigma = sep`.
Chart D already covers `epsilon |log sep| <= 1`. The remaining locus
includes

    epsilon |log sep| ~ 1/sqrt(epsilon),    1/epsilon.

The second is the kill sequence itself. The inner coordinate
`tau = epsilon log(1/sep)` equals `1/epsilon` there, so LI is not a
compact chart on that sequence. A logarithmic scale
`sigma = exp(-alpha / epsilon)` with `alpha = O(1)` makes
`epsilon |log sigma|` bounded and `sigma * kappa` unbounded, which is
the WF side of the dichotomy. Chart WL uses `Lambda = epsilon log(1/W)`.
On WS and the kill sequence, `W_e ~ epsilon^{-1} exp(-2/epsilon)` so
`Lambda_e -> 2`, but

    (W_max / W_e)^C = exp(C (Lambda_e - Lambda_max) / epsilon)

is still unbounded. Compactifying the label does not compactify the
variational factor.

**Log-chart / dichotomy verdict: fail.** The named falsifier is the same
super-small sequence, now as a proved dichotomy rather than a missing
third power of `epsilon`.

## 3. The kill sequence remains admitted

The incoming connector is the existing common-compact-connector
hypothesis of [the first-root note](HILBERT16-ROOT-SADDLE.md):
`|t_i| <= tbox < u^2/2` at one fixed `u in (0, 2)`. Its first-hit does
not see `sep`. On the kill sequence, `kappa = 1/sep` produces
`chi = 1 / r1 -> 1 / rstar`, which is order one, so the shrinking
rectangle of [coalescing §5](HILBERT16-COALESCING-CAPTURE.md) still
receives the passage. Compact-positive residence exclusion in
[the boundary-reduction note](HILBERT16-BOUNDARY-REDUCTIONS.md) requires
both endpoint `W`-labels in a fixed `[m, M]`. The outgoing label on this
sequence is not compact-positive. The sequence is therefore admitted on
the selected tube. It remains a G1 falsifier.

## 4. No existing outgoing section works

The first-root large section, the grazing cutoff `H >= gamma epsilon^3`,
and the fold-scale exit `h ~ epsilon^4` are the existing outgoing
heights. For each, `sigma = sep` on the kill sequence gives

    epsilon log(h_max / h_e) = O(epsilon |log epsilon|) + 2 / epsilon.

The W-ratio explodes. A section at `h ~ epsilon^3 sep^2` would make the
ratio order one and would lose uniform physical transversality
(`hdot -> 0`). That is a new closing map. It is not introduced.

## 5. Joint `(sep, L)` cannot cover both axes

The exact identity `2 r1 + sep + lambda1 = 0` holds whenever
`r1 = (-lambda1 - sep) / 2`. At fixed `lambda1 = -ell < 0` one cannot
have both `sep < ell/2` and `r1 < ell/4`. The super-small sequence lives
on `sep -> 0`, `r1 -> rstar > 0`. The shrinking-root sequence
`L_n = 1/n`, `lambda1 = -2` lives on `r1 -> 0`, `sep -> 2`. A joint
chart at the origin `(sep, L) -> (0, 0)` forces `lambda1 -> 0` and is
the already-named boundary-reduction corner, with its own height
normalization. SR2 rematch onto existing height / grazing / exponential
charts stays failed on the selected itinerary.

## 6. What is proved

- The exact identities `h = epsilon^3 sigma^2 eta`,
  `h1 / h2 = (sigma1 / sigma2)^2`, fold-scale `epsilon^3 sigma^2 = epsilon^4`,
  the affine leading event exponent and its vanishing second difference,
  `2 r1 + sep + lambda1 = 0`, the joint-axis exclusion, and
  `epsilon log(1/sep) = 1/epsilon` on the kill sequence.
- That the leading inner `kappa`-jet is not the G1 hole.
- That every tested scale, including log-intermediate scales, obeys the
  dichotomy on the kill sequence.
- That the kill sequence remains admitted on the selected incoming
  connector.
- That no existing outgoing section repairs WS on that sequence.
- That the two kill sequences are disjoint at fixed `lambda1 < 0`.

No physical C2 remainder, no cycle bound, and no G1 pass follow.
G4 is not opened.

## 7. G1 reassessment

G1 still requires finite charts through `sep = 0` and `r1 -> 0`, complete
first-hit on the selected connector, C2 remainders on every derivative
used downstream, and matching on overlaps.

Item 1 remains a labelled atlas plus two failed covering attempts (WF/WS
and LI/WL). Item 2 retains incoming first-hit and still has no outgoing
first-hit theorem on `L_n = 1/n`. Item 3 has no C2 remainder: the leading
inner jet being affine is not a remainder for `log D'`. Item 4 is unmatched
on `D ∩ C` at `sep = 0` and on `O ∩` anything.

**G1 does not pass.** The named remaining holes are the scale dichotomy
on `sep = exp(-1/epsilon^2)` (admitted, uncovered) and the shrinking-root
collision `L_n = 1/n`, `lambda1 = -2`. Do not open G4.

## 8. Reproduction

```bash
uv run --no-sync python benchmarks/hilbert16_next_atlas.py \
  --output artifacts/hilbert16/next_atlas.json
lake build OmnibiasAnalytic.Dynamics.Hilbert16ScaleDichotomy
```
