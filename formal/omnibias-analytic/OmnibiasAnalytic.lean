/-
Root module for the omnibias analytic (Mathlib-backed) track.

The `Check.*` modules are the sound certificate checkers (no `admit`): rational
enclosed-sign (`Check.EnclosedSign`), sum-of-squares / positivity
(`Check.Positivity`), Newton-Kantorovich / Krawczyk inequalities plus 1-D
existence on a compact interval (`Check.Kantorovich`, `Check.Kantorovich.Plant`),
named unique-zero instances (`Check.Kantorovich.Named`), rational
enclosure-trace replay (`Check.Enclosure`, `Check.Enclosure.Plant`),
compact-box residual / finite-matrix gap plants (`Check.Compact`), and
named SU(2) / SU(3) Casimir identities (`Check.Casimir`),
named polymer-coordination identities (`Check.Polymer`),
named Racah 6j identities (`Check.SixJ`), and
the Weyl-volume prefactor (`Check.HaarVolume`).
`Tower` is the Riccati / Eulerian / Hermite derivative tower (polynomial
recurrences plus `iteratedDeriv` link theorems). `Generated` is the
bridge-overwritten obligation under test.

Every module in this project contains no `admit`. The track discharges finite
rational inequalities, a unique root of a named polynomial on a compact box,
replay of a planted rational enclosure DAG, named compact-box residual /
finite-matrix gap plants, named SU(2) / SU(3) Casimir identities, and
named polymer-coordination identities, named Racah 6j identities,
and the integer Weyl-volume prefactor `6*4=24`.
It makes no continuum or asymptotic claim.
-/

import OmnibiasAnalytic.Check.EnclosedSign
import OmnibiasAnalytic.Check.Positivity
import OmnibiasAnalytic.Check.Kantorovich
import OmnibiasAnalytic.Check.Kantorovich.Plant
import OmnibiasAnalytic.Check.Kantorovich.Named
import OmnibiasAnalytic.Check.Enclosure
import OmnibiasAnalytic.Check.Enclosure.Plant
import OmnibiasAnalytic.Check.Compact
import OmnibiasAnalytic.Check.Casimir
import OmnibiasAnalytic.Check.Polymer
import OmnibiasAnalytic.Check.SixJ
import OmnibiasAnalytic.Check.HaarVolume
import OmnibiasAnalytic.Tower
import OmnibiasAnalytic.Generated
