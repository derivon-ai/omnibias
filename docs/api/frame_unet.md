# Frame-UNet (09-04)

A 1-D U-Net whose encoder raises pack order and whose decoder is
integral synthesis. Skip connections store a **band** FTC identity
and an optional **collapse** head separately; mixing them is a bug.
The founding bias collapse (`delta -> 0`) is the encoder head.
The decoder gap is the window knob. Temperature collapse
(`beta -> inf`, feasibility) does not appear. Do not conflate the
two.

Status is **shipped**. `sigma'` is not an admissible
wavelet. Frames are not orthonormal and not compactly supported.
Not Littlewood–Paley completeness. Not ImageNet. Not CCF stretch.

Homes: `omnibias.core.frame_unet`,
`omnibias.{torch,jax}.architectures.frame_unet`.

::: omnibias.core.frame_unet
    options:
      show_root_heading: false
      heading_level: 3
