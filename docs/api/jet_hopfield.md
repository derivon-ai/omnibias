# Jet-Hopfield (09-13)

Memories are germs (value plus derivatives). Retrieval scores the
contact mismatch `|u-u_μ|^2 + λ |u'-u_μ'|^2`, then applies the
modern-Hopfield softmax of `-β d^2`.

`β -> inf` is temperature collapse (hard nearest germ, feasibility)
and is labelled; the default `β` is finite. Stored profile jets may
come from founding bias collapse (`delta -> 0`) of an OMBU
dictionary. Do not conflate the two.

Status is **shipped**. Not a rewrite of vector Hopfield.
Not ImageNet retrieval. Not CCF stretch.

Homes: `omnibias.core.jet_hopfield`,
`omnibias.{torch,jax}.architectures.jet_hopfield`,
`omnibias.hopfield.{torch,jax}.ops.jet_hopfield`.

::: omnibias.core.jet_hopfield
    options:
      show_root_heading: false
      heading_level: 3
