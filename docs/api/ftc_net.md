# FTC-Net and dual-FTC training (09-03 / 09-17)

The hidden cell is the unused `integral` role
`I = S(w x + b_hi) - S(w x + b_lo)` (`S' = sigma`). The
derivative head is the fundamental theorem of that window, not a
second pack. The founding bias collapse (`delta -> 0`) is the
collapse head `I / delta`. Temperature collapse (`beta -> inf`,
feasibility) does not appear. Do not conflate the two.

Status is **shipped** for 09-03 (G1–G4 CI) and **shipped** for
09-17 (G1–G4 CI). This is a 1-D FTC identity,
not a VPINN / weak form, and not CCF stretch.
`claimed_weak_form` and `claimed_vpinn` stay false.
`theorem_prover_verified` is not asserted.

Homes: `omnibias.core.ftc`,
`omnibias.{torch,jax}.architectures.ftc_net`.

::: omnibias.core.ftc
    options:
      show_root_heading: false
      heading_level: 3
