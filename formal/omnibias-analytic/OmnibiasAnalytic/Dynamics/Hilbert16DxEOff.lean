- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
dx_e identities on lambda1 in [-4, -2]: 1 - 5/8 = 3/8, 1/(1/2) = 2,
and 11/5 + 2 = 21/5. These theorems do not cover lambda1 < -4 or
lambda1 in (-2, 0), and they do not enclose Stage C, first-hit, G1,
G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16DxEOff

/-- Left wall ``1 - 5/8 = 3/8``. -/
theorem off_wall_identity :
    (1 : ℝ) - 5 / 8 - 3 / 8 = 0 := by
  ring

/-- Ratio ``1 / (1/2) = 2``. -/
theorem off_u_identity :
    (1 : ℝ) / (1 / 2) - 2 = 0 := by
  ring

/-- Chi cap ``11/5 + 2 = 21/5``. -/
theorem off_chi_identity :
    (11 / 5 : ℝ) + 2 - 21 / 5 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16DxEOff
