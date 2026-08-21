# Sliced optimal transport (03-04)

A tempered-activation mixture has a closed-form CDF, so one-dimensional
`W_1` is an exact integral of `|F-G|` (sign-change roots plus
softplus antiderivatives). Status is **gated**, not shipped. G1–G6
are CI-gated.

Mixture components come from the **founding bias collapse**
(`delta -> 0` to `sigma'`). Temperature collapse (`beta -> inf`,
feasibility) does not appear. Do not conflate the two. Distances are
**exact per slice** and **sampled over directions** (`O(1/L)`); the
method is not sample-free. Sliced Wasserstein is not Wasserstein.

Home: `omnibias.measure.transport`. No new package.

## API

::: omnibias.measure.transport
    options:
      show_root_heading: false
      heading_level: 3
