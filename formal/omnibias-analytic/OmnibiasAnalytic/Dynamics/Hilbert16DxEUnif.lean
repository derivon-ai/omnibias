/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line uniform-in-chi dx_e identities: extra coefficient
(3/16)*(1/2) = 3/32, written c = 1/16 weaker by 1/32, and
threshold lift (3/32)*9 = 27/32. These theorems do not enclose
Stage C, first-hit, dx_e off the kill line, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16DxEUnif

/-- Extra chi-coefficient ``(3/16)*(1/2) = 3/32``. -/
theorem extra_half_identity :
    (3 / 16 : ℝ) * (1 / 2) - 3 / 32 = 0 := by
  ring

/-- Written decay ``3/32 - 1/16 - 1/32 = 0``. -/
theorem c_weaker_identity :
    (3 / 32 : ℝ) - 1 / 16 - 1 / 32 = 0 := by
  ring

/-- Threshold lift ``(3/32)*9 = 27/32``. -/
theorem lift_nine_identity :
    (3 / 32 : ℝ) * 9 - 27 / 32 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16DxEUnif
