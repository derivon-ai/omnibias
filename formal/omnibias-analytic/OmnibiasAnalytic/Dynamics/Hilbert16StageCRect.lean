/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line Stage-C continuation-rectangle identities: 1+1=2, 2*2=4,
and (1/16)*(1/4)=1/64. These theorems do not enclose first-hit, C2,
G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16StageCRect

/-- T-wall ``1 + 1 = 2`` at hmax. -/
theorem hmax_two_identity :
    (1 : ℝ) + 1 - 2 = 0 := by
  ring

/-- Left-wall ``2 * 2 = 4``. -/
theorem twice_two_identity :
    (2 : ℝ) * 2 - 4 = 0 := by
  ring

/-- Right-wall ``(1/16) * (1/4) = 1/64``. -/
theorem amin_eps_identity :
    (1 : ℝ) / 16 * (1 / 4) - 1 / 64 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16StageCRect
