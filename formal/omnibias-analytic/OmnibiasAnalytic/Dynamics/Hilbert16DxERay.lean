- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
dx_e identities for every lambda1 <= -2: X = 2*rstar equals 2 at
rstar = 1, (3/8)/(2*rstar)*rstar = 3/16, and 11/5 + 2 = 21/5.
These theorems do not cover lambda1 in (-2, 0), Stage C, first-hit,
G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16DxERay

/-- Residence bound ``2 * 1 = 2``. -/
theorem ray_x_identity :
    (2 : ℝ) * 1 - 2 = 0 := by
  ring

/-- Coefficient ``(3/8) / 2 = 3/16``. -/
theorem ray_coeff_identity :
    ((3 / 8 : ℝ) / 2) - 3 / 16 = 0 := by
  ring

/-- Chi cap ``11/5 + 2 = 21/5``. -/
theorem ray_chi_identity :
    (11 / 5 : ℝ) + 2 - 21 / 5 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16DxERay
