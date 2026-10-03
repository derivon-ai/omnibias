/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Exact weighted-section obstruction identities.  They show finite weighted
derivatives but singular ordinary coefficient derivatives at D ∩ C, and a
vanishing chart-O interface speed.  They do not prove G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16WeightedSection

/-- The scalar hit time has `rate*q*tau_q = -1`. -/
theorem weightedHitFirst
    (rate q : ℝ)
    (hrate : rate ≠ 0)
    (hq : q ≠ 0) :
    rate * q * (-(1 / (rate * q))) + 1 = 0 := by
  field_simp

/-- The scalar hit time has `rate*q^2*tau_qq = 1`. -/
theorem weightedHitSecond
    (rate q : ℝ)
    (hrate : rate ≠ 0)
    (hq : q ≠ 0) :
    rate * q ^ 2 * (1 / (rate * q ^ 2)) - 1 = 0 := by
  field_simp

/-- On `r1=q`, `r2=2-q`, the chart-O interface speed is proportional to `q`. -/
theorem originWeightedSpeed
    (q theta eta0 : ℝ) :
    (4 - 2 * q) * (q * (1 + theta) * eta0)
      - 2 * (q * (2 - q)) * (1 + theta) * eta0 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16WeightedSection
