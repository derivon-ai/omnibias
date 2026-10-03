- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line identities for sep * S_pre on every sep in (0, 1]:
2^48 * 2^{-48} = 1, 4/(1/2) = 8, and (3/16)(4 - 11/5) = 27/80.
These theorems do not enclose dx_e off the kill line, a uniform-in-chi
bound, Stage C, first-hit, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16SepSpre

/-- Tail node ``2^48 * 2^{-48} = 1``. -/
theorem spre_tail_identity :
    (2 : ℝ) ^ 48 * (1 / 2 ^ 48) - 1 = 0 := by
  ring

/-- Wall chi ``4 / (1/2) = 8``. -/
theorem spre_chi_identity :
    (4 : ℝ) / (1 / 2) - 8 = 0 := by
  ring

/-- Net floor ``(3/16) * (4 - 11/5) = 27/80``. -/
theorem spre_net_identity :
    (3 / 16 : ℝ) * (4 - 11 / 5) - 27 / 80 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16SepSpre
