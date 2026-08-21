# Kantorovich-accepted Newton (08-04)

A Gauss–Newton or cubic-Newton trial is **legal** only when the radii
polynomial returns a nonempty unique-zero ball around the trial point.
Empty is a valid reject: do not train through it.

Status is **gated**, not shipped. G1–G3 are CI-gated. The ball is of a
**finite residual map**, never a continuum PDE solution. Sound-enclosure
tier: `theorem_prover_verified` is not asserted. Not CCF stretch and not
Navier–Stokes regularity. Bias collapse (`delta -> 0`) may tighten `DF`
via exact `sigma'`; this policy does not require it. See theory spec
08-04.

Stack: direction (GN) → length (03-12) → this accept/reject. Distinct
from 08-09 (input-output Lipschitz). The radii-polynomial algebra is
documented with the verified backend in [core.md](core.md).

## PyTorch twin

::: omnibias.torch.optim_kantorovich
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.optim_kantorovich
    options:
      show_root_heading: false
      heading_level: 3
