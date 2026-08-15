# Hierarchical pack tree (02-07)

Near/far split for large 1-D offset banks. Far-field is a truncation
with a bound that never undercovers. `eta = 0` is bit-identical to the
dense sum (same summands, same order). 1-D offset axis only; no 2-D/3-D
FMM.

G1 (`eta=0`), G2 (bound never undercovers), G4 target accuracy, and G5
parity are CI-gated. G3 complexity is smoke-recorded, not in CI
`all_passed`. Status is **gated**, not shipped. See theory spec 02-07.

## Core algebra

::: omnibias.core.hierarchy
    options:
      show_root_heading: false
      heading_level: 3
