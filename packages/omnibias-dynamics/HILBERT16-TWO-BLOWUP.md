# Two-blow-up covering in the central-strip coordinates

This companion continues [the saddle-node note](HILBERT16-SADDLE-NODE.md)
by replacing a single scale `sigma` with a pair of charts in the existing
coordinates of [the boundary-reduction note](HILBERT16-BOUNDARY-REDUCTIONS.md):

    u = V / epsilon,    W = h^epsilon / epsilon.

It reuses the incoming connector `|t_i| <= tbox < u^2/2` and the large
outgoing section of [the first-root note](HILBERT16-ROOT-SADDLE.md). It
does not invent a new closing map and does not apply Huzak–Kristiansen
Theorem 2.4 as a C2 or cyclicity statement.

Discovery, named `lim` paths, and symbolic reduction sit upstream: they
propose the identities below. They do not replace a continuum remainder.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. Exact W-outgoing identity

On `h > 0` and `epsilon > 0`, `W = h^epsilon / epsilon` is an ordinary
change of variables, so `h^epsilon = epsilon W`. Therefore

    (h_max / h_e)^{C epsilon} = (W_max / W_e)^C

exactly, for any positive `C`. The first-root outgoing variational factor
is a **power of a W-ratio**, not a leftover `epsilon |log h_e|` that has
to be estimated separately. The central-strip identity
`d(log W) / d tau = -u` is the exact field identity of boundary-reduction
§1. These two facts are the whole gain of moving to `(u, W)`.

They do not by themselves bound either `W_max / W_e` or the event factor
`exp(C sigma kappa)`.

## 2. Chart WF (W-fold)

Blow up the fold in `u` at scale `sigma = sqrt(epsilon)`, as in the
saddle-node fold chart. The physical exit height is
`h_e = O(epsilon^4)`, independent of `sep`. Then

    W_e = h_e^epsilon / epsilon = O(epsilon^{4 epsilon - 1}).

For a selected outgoing height `h_max` of order `epsilon^3` or of order
one, `W_max / W_e -> 1` as `epsilon -> 0`, uniformly in `sep`, including
`sep = 0`. Chart WF therefore absorbs the outgoing half of the
saddle-node tension.

The event identity is unchanged. On a `chi = O(1)` locus,
`sigma * kappa = (sqrt(epsilon) / sep) * chi * r1`. The kill sequence

    sep_n = exp(-1 / epsilon_n^2),    kappa_n = 1 / sep_n,
    lambda1 = -3,    L_n = (9 - sep_n^2) / 4

still makes `sigma * kappa` unbounded. Chart WF covers `sep = 0` and the
restricted tail `kappa = O(epsilon^{-1/2})`. It does not cover that
sequence.

## 3. Chart WS (W-separation)

On `chi = O(1)` take the separation-scale inner box `sigma = sep`, so
`sigma * kappa` stays of order one. The exit height is
`h_e = O(epsilon^3 sep^2)`, hence

    W_e = epsilon^{3 epsilon - 1} * sep^{2 epsilon}
        = epsilon^{3 epsilon - 1} * exp(2 epsilon log sep).

On the kill sequence, `epsilon log sep = -1 / epsilon`, so
`sep^{2 epsilon} = exp(-2 / epsilon)` and `W_e -> 0` faster than any
power of `epsilon`. The W-ratio `(W_max / W_e)^C` is then unbounded.
Moving to `W` does **not** remove chart D’s outgoing blow-up on that
sequence. It rewrites it as a W-ratio and the ratio still explodes.

No overlap matching can send the kill sequence into WF: WF forbids
`kappa = 1 / sep`. No overlap matching can send it into WS: WS forbids
the outgoing W-ratio.

## 4. Two-scale product and C2

The algebraic product of the two event factors is the identity

    (sigma_fold * kappa) * (sigma_sep / sigma_fold) = sigma_sep * kappa.

It does not produce a third scale that is both `>= sqrt(epsilon)` and
`<= sep` when `sep << sqrt(epsilon)`. Differentiating `log D'` a second
time again contracts residence time against `epsilon sigma`. Because the
first derivative is already unbounded on the kill sequence in every
chart of §§2–3, no C2 remainder is claimed.

Huzak–Kristiansen’s two-step blow-up applies to compact positive labels
with a nonvanishing weighted drift. The present itinerary is the
complementary small-label tube. Their Theorem 2.4 is not used.

**Two-blow-up C2 verdict: fail.** The named falsifier is the same
super-small sequence as coalescing §5.5 and saddle-node §3, now written
in `(u, W)`.

## 5. What is proved

- The exact identities `epsilon W = h^epsilon` and
  `(h_max / h_e)^{C epsilon} = (W_max / W_e)^C`.
- That WF makes the outgoing W-ratio uniform in `sep`, including
  `sep = 0`.
- That WS makes `sigma * kappa` of order one on `chi = O(1)`.
- That no pair of these charts covers the kill sequence.

No physical C2 remainder, no cycle bound, and no G1 pass follow.
The `L -> 0` layer remains open and is treated in
[the shrinking-root note](HILBERT16-SHRINKING-ROOT.md) §7.
The [next-atlas note](HILBERT16-NEXT-ATLAS.md) tests logarithmic
charts LI/WL and records the same kill sequence as a scale dichotomy:
no positive `sigma` bounds both `sigma * kappa` and the W-ratio to an
existing `epsilon^N` section. The [chart-cell ledger](HILBERT16-CHART-CELLS.md)
records WF, WS, LI, WL, and the named kill sequences; a complete list
of labels is not G1.

## 6. Reproduction

```bash
uv run --no-sync python benchmarks/hilbert16_two_blowup.py \
  --output artifacts/hilbert16/two_blowup.json
lake build OmnibiasAnalytic.Dynamics.Hilbert16TwoBlowup
```
