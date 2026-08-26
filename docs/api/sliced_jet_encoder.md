# Sliced-jet encoder (09-28)

A holistic encoder whose tokens are 1-D scans along named directions
`w`, optionally jetted along the scan axis. founding bias collapse
(`delta -> 0`) supplies that jet. Soft selection `beta -> inf` is
temperature collapse (feasibility) and is labelled.

A sparse readout must name its energy. Tokens are slices, not
patches. Not a ViT. Not ImageNet. Not `R^D` equivariance. Not CCF
stretch. Status is **shipped**.

Homes: `omnibias.core.sliced_jet`,
`omnibias.{torch,jax}.architectures.sliced_jet`.

::: omnibias.core.sliced_jet
    options:
      show_root_heading: false
      heading_level: 3
