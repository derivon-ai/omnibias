/-
Root module for the omnibias analytic (Mathlib-backed) track.

The `Check.*` modules are the sound certificate checkers (no `admit`): rational
enclosed-sign (`Check.EnclosedSign`), sum-of-squares / positivity
(`Check.Positivity`), Newton-Kantorovich / Krawczyk inequalities plus 1-D
existence on a compact interval (`Check.Kantorovich`, `Check.Kantorovich.Plant`),
named unique-zero instances (`Check.Kantorovich.Named`), and rational
enclosure-trace replay (`Check.Enclosure`, `Check.Enclosure.Plant`).
`Tower` is the Riccati / Eulerian / Hermite derivative tower (polynomial
recurrences plus `iteratedDeriv` link theorems). `Generated` is the
bridge-overwritten obligation under test.

Every module in this project contains no `admit`. The track discharges finite
rational inequalities, a unique root of a named polynomial on a compact box,
and replay of a planted rational enclosure DAG. It makes no continuum or
asymptotic claim.
-/

import OmnibiasAnalytic.Check.EnclosedSign
import OmnibiasAnalytic.Check.Positivity
import OmnibiasAnalytic.Check.Kantorovich
import OmnibiasAnalytic.Check.Kantorovich.Plant
import OmnibiasAnalytic.Check.Kantorovich.Named
import OmnibiasAnalytic.Check.Enclosure
import OmnibiasAnalytic.Check.Enclosure.Plant
import OmnibiasAnalytic.Tower
import OmnibiasAnalytic.Generated
