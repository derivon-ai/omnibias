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
the Weyl-volume prefactor (`Check.HaarVolume`),
and finite rational convergence-ledger margins (`Check.ConvergenceLedger`),
and the locked admissible-stress cone (`Check.StressCone`).
`Tower` is the Riccati / Eulerian / Hermite derivative tower (polynomial
recurrences plus `iteratedDeriv` link theorems). `Generated` is the
bridge-overwritten obligation under test.

Every module in this project contains no `admit`. The track discharges finite
rational inequalities, a unique root of a named polynomial on a compact box,
replay of a planted rational enclosure DAG, named compact-box residual /
finite-matrix gap plants, named SU(2) / SU(3) Casimir identities, and
named polymer-coordination identities, named Racah 6j identities,
and the integer Weyl-volume prefactor `6*4=24`.
The Dynamics modules additionally check real-variable derivative and zero-count
implications under explicit analytic hypotheses. They do not establish full
physical passage estimates, global cycle capture, or Hilbert XVI.
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
import OmnibiasAnalytic.Check.ConvergenceLedger
import OmnibiasAnalytic.Check.StressCone
import OmnibiasAnalytic.Tower
import OmnibiasAnalytic.RealizationReplay
import OmnibiasAnalytic.RealizationAlgebra
import OmnibiasAnalytic.Dynamics.Hilbert16Rolle
import OmnibiasAnalytic.Dynamics.Hilbert16Parabola
import OmnibiasAnalytic.Dynamics.Hilbert16Resonance
import OmnibiasAnalytic.Dynamics.Hilbert16ReturnMap
import OmnibiasAnalytic.Dynamics.Hilbert16Scale
import OmnibiasAnalytic.Dynamics.Hilbert16ChiScale
import OmnibiasAnalytic.Dynamics.Hilbert16SaddleNode
import OmnibiasAnalytic.Dynamics.Hilbert16TwoBlowup
import OmnibiasAnalytic.Dynamics.Hilbert16ScaleDichotomy
import OmnibiasAnalytic.Generated
