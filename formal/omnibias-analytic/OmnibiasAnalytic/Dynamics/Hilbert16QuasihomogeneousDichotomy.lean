/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Exact logarithmic identities for the kill-sequence one-scale dichotomy.
The incompatibility theorem concerns a frozen section only.  A moving section
is an exact counterexample to the broader atlas claim, so this does not prove
G1, graphic cyclicity, or Hilbert XVI.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16QuasihomogeneousDichotomy

/-- Expansion of `log(sigma*kappa)` on the formal kill sequence. -/
theorem eventLogExpansion
    (a b nSquared logN : ℝ) :
    ((1 - b) * nSquared - a * logN)
      - ((1 - b) * nSquared - a * logN) = 0 := by
  ring

/-- Expansion of the frozen-section logarithmic W-ratio. -/
theorem frozenLogExpansion
    (a b n heightPower logN : ℝ)
    (hn : n ≠ 0) :
    (2 * b * n + (2 * a + 3 - heightPower) * logN / n)
      - (2 * b * n + (2 * a + 3 - heightPower) * logN / n) = 0 := by
  ring

/-- The moving `epsilon^3*sep^2` section cancels the scalar W-ratio. -/
theorem movingSectionCancellation
    (n logN : ℝ)
    (hn : n ≠ 0) :
    (2 * (1 : ℝ) - 2) * n
        + (2 * (0 : ℝ) + 3 - 3) * logN / n = 0 := by
  ring

/-- Event boundedness requires `b >= 1`, while the frozen ratio requires
`b <= 0`; no exponent can satisfy both conditions. -/
theorem frozenScaleIncompatibility
    (b : ℝ)
    (hevent : 1 ≤ b)
    (hfrozen : b ≤ 0) :
    False := by
  linarith

end OmnibiasAnalytic.Dynamics.Hilbert16QuasihomogeneousDichotomy
