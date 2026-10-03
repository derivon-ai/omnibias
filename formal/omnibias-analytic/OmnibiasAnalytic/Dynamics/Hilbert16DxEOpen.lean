- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
dx_e identities for lambda1 in (-3/2, 0): the wall vanishes at
sep = (8/5) rstar, 11/5 + 5 = 36/5, and 3/16 - 1/20 = 11/80.
These theorems do not cover Stage C, first-hit, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16DxEOpen

/-- Wall ``1 - (5/8)*(8/5) = 0``. -/
theorem open_wall_identity :
    (1 : ℝ) - (5 / 8) * (8 / 5) = 0 := by
  ring

/-- Chi cap ``11/5 + 5 = 36/5``. -/
theorem open_chi_identity :
    (11 / 5 : ℝ) + 5 - 36 / 5 = 0 := by
  ring

/-- Net floor ``3/16 - 1/20 = 11/80``. -/
theorem open_gap_identity :
    (3 / 16 : ℝ) - 1 / 20 - 11 / 80 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16DxEOpen
