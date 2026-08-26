# Enclosure Collapse and the Width Law (01-14)

Enclosure Collapse is the `width -> 0` limit of a **sound enclosure**.
The output is a point **plus a proof**, or `Inconclusive`. It is **not**
a derivative and **not** a 0/1 step. Status is **shipped**.
G1–G6 are package-test gated.

The founding **bias collapse** (`delta -> 0`) yields `sigma^(K-1)`.
Temperature collapse (`beta -> inf`, feasibility) yields a 0/1
indicator. Do not conflate the three.

`lo` and `hi` are not two biases. The integral window
`S(z+b_hi)-S(z+b_lo)` and conformal slabs (04-02) are different
objects. Forcing `lo = hi` by clamping is unsound.

Homes: `omnibias.core.verified.enclosure_collapse` (Width Law,
`diagnose_width`) and `omnibias.verify.enclosure_collapse` (six
`squeeze_*` wrappers). No new package. `WidthBudget` is reused, not
forked. `theorem_prover_verified` defaults `False`.

## Algebra

::: omnibias.core.verified.enclosure_collapse
    options:
      show_root_heading: false
      heading_level: 3

## Squeeze facade

::: omnibias.verify.enclosure_collapse
    options:
      show_root_heading: false
      heading_level: 3
