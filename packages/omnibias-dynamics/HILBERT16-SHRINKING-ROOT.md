# Shrinking-root chart for `r1 -> 0`

This companion continues the [χ-atlas](HILBERT16-COALESCING-CAPTURE.md)
and the [saddle-node note](HILBERT16-SADDLE-NODE.md) in the complementary
direction `L -> 0` at fixed negative first-root sign. It reuses the
incoming connector `|t_i| <= tbox < u^2/2` and the large outgoing
section of [the first-root note](HILBERT16-ROOT-SADDLE.md). It does not
invent a new closing map.

The saddle-node increment recorded a scale tension on the super-small
separation sequence at `L >= Lmin`. That failure leaves `L -> 0` open, so
this note proceeds. It does **not** claim a first-hit theorem for every
height, a C2 remainder, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. Exact shrinking first root

Keep `lambda1 <= -lmin < 0` first, with `L > 0` small enough that
`sep = sqrt(lambda1^2 - 4 L)` is real. Then

    r1 = (-lambda1 - sep) / 2 = 2 L / (|lambda1| + sep) -> 0

as `L -> 0`, while

    sep -> |lambda1| >= lmin,    r2 = (-lambda1 + sep) / 2 -> |lambda1|.

The identity `r1 * (|lambda1| + sep) = 2 L` is polynomial once
`sep^2 = lambda1^2 - 4 L` is substituted. The outgoing saddle of the
first-root note sits at `x = r_eps = r1 + O(epsilon)`, hence at

    V = -epsilon * r_eps -> 0

in the physical chart. The uniform wall `a_min > 0` used by charts F and D
fails: any fixed `d > 0` eventually exceeds `r1`.

The joint corner `lambda1 -> 0` together with `L -> 0` is the
boundary-reduction weighted chart
`lambda1 = kappa_BR * l1`, `lambda0 = kappa_BR^2 * l0`. That argument
keeps its own height normalization. It is not a corollary of the
rescaling below.

## 2. Radial rescaling

Set `x = r1 * xi`. The first-root wall is the `O(1)` interval `xi in [0, 1]`.
Exactly,

    B_-(x) = (x - r1) * (x - r2) = r1^2 * (xi - 1) * (xi - r2 / r1),
    r2 / r1 = (|lambda1| + sep)^2 / (4 L) -> infinity.

The second root leaves every compact `xi` box. The χ-atlas of the
coalescing note used `sep` as the small quantity at `L >= Lmin`, with
matching coordinate `chi = (sep / r1) * kappa`. Here `sep / r1` is of
order `1 / L` and diverges. On a `chi = O(1)` locus one would need
`kappa = O(r1 / sep) = O(L)`, a vanishing slow time, not the first-root
tail. The overlap with that atlas is therefore empty for `L << Lmin` at
fixed `lambda1 <= -lmin`. The no-root `Delta` chart lives on
`Delta = 4 L - lambda1^2 > 0` and is likewise disjoint from this
first-root sign.

A compact `(xi, eta)` description can still be written on
`xi in [0, 1 + theta]` for fixed `theta in (0, 1)`. It does not restore a
uniform physical distance from the centre.

## 3. Incoming first-hit, outgoing collision

The incoming connector is unchanged. For the selected tube
`|t_i| <= tbox < u^2/2` at one fixed `u in (0, 2)`, the ordinary
positive-height fast connector to `V = u` has uniform positive height and
a uniformly nonzero derivative, independently of `r1`. That is the
existing common-compact-connector hypothesis of the first-root note,
section 1. This note therefore retains incoming first-hit on that tube.
It does not assert that every centre height is admitted.

Outgoing continuation is the obstruction. The first-root saddle rectangle
requires a fixed `d > 0` small compared with the lower bounds on `r1` and
`r2 - r1`, and then `a = r_eps - d >= a_min > 0`. As `r1 -> 0` no such
uniform `a_min` exists. After the `xi` rescaling the exit sits at
`x_e = O(r1)`, so

    V_e = -epsilon * x_e = O(epsilon * r1),    T_e = O(epsilon^2 r1^2).

The first-root continuation bootstrap uses `|V| >= epsilon * a_min` to
keep a definite margin from the centre `V = 0`, and a fixed negative-`V`
rectangle reaching a selected large outgoing section near its zero label.
When `a_min` must shrink with `r1`, that margin vanishes: the saddle
itself collides with the centre on the scale `epsilon * r1`. The estimate
`T - h = O(epsilon)` on a rectangle whose left wall stays a fixed
distance from `V = 0` is then unavailable.

The selected large outgoing section is a fixed physical object. Reaching
it from `V_e = O(epsilon * r1)` requires crossing a vanishing corridor
around the centre. The first-root argument does not supply a uniform
transversality margin for that crossing.

## 4. Named falsifier

The sequence of coalescing §7,

    L_n = 1 / n,    lambda1 = -2,    sep_n -> 2,    r1_n -> 0,

is admitted by the first-root coefficient inequalities except
`L >= Lmin`. Along it, `V_saddle,n = -epsilon * r1_n -> 0` at every fixed
`epsilon`. Continuation from that saddle to the large outgoing section
loses every margin that the first-root note treated as uniform.
This is an SR falsifier, not a counterexample to finite cyclicity.

No C2 remainder is claimed. Differentiating the outgoing labels would
again use `x_e >= a_min` and the same continuation factor
`(h_max / h_e)^{C epsilon}`. Both fail to stay uniform.

**Shrinking-root verdict: fail.** Incoming first-hit on the selected
connector is retained. Outgoing first-hit of the large section, a uniform
`a_min`, and C2 remainders are not obtained.

## 5. What is proved

- The exact shrinking-root identities `r1 = 2 L / (|lambda1| + sep)` and
  `r1 * (|lambda1| + sep) = 2 L`.
- That the χ-atlas and the no-root `Delta` chart do not overlap this
  layer at fixed `lambda1 <= -lmin`.
- That the selected incoming tube still first-hits `V = u`.
- That the outgoing saddle collides with the centre along `L_n = 1 / n`,
  `lambda1 = -2`.

No cycle bound and no G1 pass follow.

## 7. SR2 rematch onto existing height charts

The two-blow-up increment left `L -> 0` open. This section tests membership
of `L_n = 1 / n`, `lambda1 = -2` in the **existing** height, grazing, and
exponential coefficient sets. No new closing map is introduced.
Recall `L = -lambda0` from [the first-root note](HILBERT16-ROOT-SADDLE.md).

- [Height comparison](HILBERT16-HEIGHT-COMPARISON.md) requires
  `0 <= lambda1 <= L1`. Here `lambda1 = -2`. The sequence is not admitted.
- [Exponential passage](HILBERT16-EXPONENTIAL-PASSAGE.md) requires
  `L = -lambda0 ∈ [c, L0]` with a fixed `c > 0`. Along `L_n = 1 / n` one
  eventually has `L_n < c`. The sequence is not admitted uniformly.
- [Grazing](HILBERT16-GRAZING-PASSAGE.md) (1.1) allows
  `-L0 <= lambda0 <= 0` and `|lambda1| <= L0`. For `L0 >= 2` the
  coefficients of each finite `n` lie in that box, on the height band
  `H >= gamma epsilon^3`. That is a different itinerary: it starts at
  `V = 0` with a center-height cutoff, not at the selected small-label
  connector with `kappa -> infinity`. Grazing also excludes the joint
  degeneration `lambda0 -> 0` with `H / epsilon^3 -> 0`. The selected
  first-root tube is not that band.

**SR2 rematch verdict: fail.** Incoming first-hit at `V = u` is retained.
Outgoing first-hit of the selected large first-root section is not recovered
by any existing height theorem along `L_n = 1 / n`, `lambda1 = -2`. The
joint corner `lambda1 -> 0` and `L -> 0` stays the boundary-reduction
weighted chart. The exact identity `2 r1 + sep + lambda1 = 0` of
[the next-atlas note](HILBERT16-NEXT-ATLAS.md) shows this axis is disjoint
from `sep -> 0` at fixed `lambda1 < 0`; a joint `(sep, L)` chart does not
absorb the sequence.

## 8. Reproduction

The algebraic identities are replayed by
[the saddle-node benchmark](../../benchmarks/hilbert16_saddle_node.py)
and by
[Hilbert16SaddleNode.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16SaddleNode.lean).
Those checks do not prove outgoing continuation.

```bash
uv run --no-sync python benchmarks/hilbert16_saddle_node.py \
  --output artifacts/hilbert16/saddle_node.json
uv run --no-sync python benchmarks/hilbert16_two_blowup.py \
  --output artifacts/hilbert16/two_blowup.json
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_saddle_node.py \
  packages/omnibias-dynamics/tests/test_two_blowup.py -q
```
