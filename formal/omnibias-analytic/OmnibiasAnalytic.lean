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
import OmnibiasAnalytic.Check.Hilbert16Cyclicity
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
import OmnibiasAnalytic.Dynamics.Hilbert16WeightedSection
import OmnibiasAnalytic.Dynamics.Hilbert16QuasihomogeneousDichotomy
import OmnibiasAnalytic.Dynamics.Hilbert16LNFormatBarrier
import OmnibiasAnalytic.Dynamics.Hilbert16AbelianReturnTransfer
import OmnibiasAnalytic.Dynamics.Hilbert16BautinJetBarrier
import OmnibiasAnalytic.Dynamics.Hilbert16SonglingPrecision
import OmnibiasAnalytic.Dynamics.Hilbert16PartAPolygonSOS
import OmnibiasAnalytic.Dynamics.Hilbert16LNCell
import OmnibiasAnalytic.Dynamics.Hilbert16EntryExit
import OmnibiasAnalytic.Dynamics.Hilbert16StageB
import OmnibiasAnalytic.Dynamics.Hilbert16StageA
import OmnibiasAnalytic.Dynamics.Hilbert16ChiB
import OmnibiasAnalytic.Dynamics.Hilbert16DxELeading
import OmnibiasAnalytic.Dynamics.Hilbert16DxEUnif
import OmnibiasAnalytic.Dynamics.Hilbert16StageC
import OmnibiasAnalytic.Dynamics.Hilbert16StageCExit
import OmnibiasAnalytic.Dynamics.Hilbert16StageCTh
import OmnibiasAnalytic.Dynamics.Hilbert16StageCGap
import OmnibiasAnalytic.Dynamics.Hilbert16StageCEnv
import OmnibiasAnalytic.Dynamics.Hilbert16StageCIf
import OmnibiasAnalytic.Dynamics.Hilbert16StageCInt
import OmnibiasAnalytic.Dynamics.Hilbert16StageCLo
import OmnibiasAnalytic.Dynamics.Hilbert16StageCK
import OmnibiasAnalytic.Dynamics.Hilbert16StageCBoot
import OmnibiasAnalytic.Dynamics.Hilbert16StageCRect
import OmnibiasAnalytic.Dynamics.Hilbert16StageCHit
import OmnibiasAnalytic.Dynamics.Hilbert16StageCSec
import OmnibiasAnalytic.Dynamics.Hilbert16StageCOneshot
import OmnibiasAnalytic.Dynamics.Hilbert16StageCOneshotEps
import OmnibiasAnalytic.Dynamics.Hilbert16StageCEpsSpan
import OmnibiasAnalytic.Dynamics.Hilbert16StageCOrigin
import OmnibiasAnalytic.Dynamics.Hilbert16StageCOriginSpan
import OmnibiasAnalytic.Dynamics.Hilbert16StageCOriginIface
import OmnibiasAnalytic.Dynamics.Hilbert16StageCOriginNear
import OmnibiasAnalytic.Dynamics.Hilbert16StageCOriginX32
import OmnibiasAnalytic.Dynamics.Hilbert16StageCCompare
import OmnibiasAnalytic.Dynamics.Hilbert16StageCUniform
import OmnibiasAnalytic.Dynamics.Hilbert16StageCInterface
import OmnibiasAnalytic.Dynamics.Hilbert16SepSpre
import OmnibiasAnalytic.Dynamics.Hilbert16DxEOff
import OmnibiasAnalytic.Dynamics.Hilbert16DxERay
import OmnibiasAnalytic.Dynamics.Hilbert16DxENear
import OmnibiasAnalytic.Dynamics.Hilbert16DxEOpen
import OmnibiasAnalytic.Dynamics.Hilbert16FoldLeading
import OmnibiasAnalytic.Dynamics.Hilbert16ShrinkingRoot
import OmnibiasAnalytic.Dynamics.Hilbert16CanonicalZeta
import OmnibiasAnalytic.Dynamics.Hilbert16FoldZeta
import OmnibiasAnalytic.Dynamics.Hilbert16PhysicalC2
import OmnibiasAnalytic.Dynamics.Hilbert16ZXGap
import OmnibiasAnalytic.Dynamics.Hilbert16ZVBound
import OmnibiasAnalytic.Dynamics.Hilbert16ZSlowV
import OmnibiasAnalytic.Dynamics.Hilbert16FoldZX
import OmnibiasAnalytic.Dynamics.Hilbert16OutgoingCorridor
import OmnibiasAnalytic.Dynamics.Hilbert16PostCorridor
import OmnibiasAnalytic.Dynamics.Hilbert16HeightEnvelope
import OmnibiasAnalytic.Dynamics.Hilbert16QRatioC2
import OmnibiasAnalytic.Dynamics.Hilbert16KZetaRemainder
import OmnibiasAnalytic.Dynamics.Hilbert16KillZeta
import OmnibiasAnalytic.Dynamics.Hilbert16CancelledN
import OmnibiasAnalytic.Dynamics.Hilbert16HeightMix
import OmnibiasAnalytic.Dynamics.Hilbert16OrbitTh
import OmnibiasAnalytic.Dynamics.Hilbert16ThIntegral
import OmnibiasAnalytic.Dynamics.Hilbert16VhOrbit
import OmnibiasAnalytic.Dynamics.Hilbert16EOutSection
import OmnibiasAnalytic.Dynamics.Hilbert16EOutEps
import OmnibiasAnalytic.Dynamics.Hilbert16EOutSpeed
import OmnibiasAnalytic.Dynamics.Hilbert16ESigmaSpeed
import OmnibiasAnalytic.Dynamics.Hilbert16ESigmaIn
import OmnibiasAnalytic.Dynamics.Hilbert16ESigmaHit
import OmnibiasAnalytic.Dynamics.Hilbert16ESigmaFrom0
import OmnibiasAnalytic.Dynamics.Hilbert16ESigmaUnif
import OmnibiasAnalytic.Dynamics.Hilbert16ESigmaWall
import OmnibiasAnalytic.Dynamics.Hilbert16ESigmaBox
import OmnibiasAnalytic.Dynamics.Hilbert16ESigmaSpan
import OmnibiasAnalytic.Dynamics.Hilbert16ESigmaPack
import OmnibiasAnalytic.Dynamics.Hilbert16ESigmaEps
import OmnibiasAnalytic.Dynamics.Hilbert16ESigmaOneshot
import OmnibiasAnalytic.Dynamics.Hilbert16ESigmaOneshotEps
import OmnibiasAnalytic.Dynamics.Hilbert16ESigmaEpsSpan
import OmnibiasAnalytic.Dynamics.Hilbert16ESigmaEpsLo
import OmnibiasAnalytic.Generated
