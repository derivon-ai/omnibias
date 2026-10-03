/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Cleared two-root I-map derivative and the first-root wall sign. These
theorems do not bound height-section first-hit, uniform a_min, G1, G4,
or Hilbert XVI.
-/

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16OutgoingCorridor

/-- Cleared derivative of `J = -r₁ log|x-r₁| + r₂ log|x-r₂|`. -/
theorem J_numerator_identity (x r1 r2 : ℝ) :
    -r1 * (x - r2) + r2 * (x - r1) - (r2 - r1) * x = 0 := by
  ring

/-- The first-root wall `a = r₁ - d` is negative once `r₁ < d`. -/
theorem a_wall_fails {r1 d : ℝ} (h : r1 < d) : r1 - d < 0 := by
  linarith

end OmnibiasAnalytic.Dynamics.Hilbert16OutgoingCorridor
