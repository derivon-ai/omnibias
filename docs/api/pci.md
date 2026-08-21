# Proof-carrying forward (09-24)

A TM forward that returns `(y_mid, box)`. Distinct from 08-09, which
filters a parameter step. The founding bias collapse (`delta -> 0`)
supplies the polynomial part. Temperature collapse (`beta -> inf`,
feasibility) does not appear. Do not conflate the two.

Status is **gated**, not shipped. Lean flags stay false unless a
later PR attaches a genuine `lake build`. Not ImageNet. Not CCF
stretch.

Home: `omnibias.verify._core.pci`.

::: omnibias.verify._core.pci
    options:
      show_root_heading: false
      heading_level: 3
