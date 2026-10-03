/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Actual-versus-comparison T_h gap: the difference splits as
(k-1) + (q - C ε(T+ε²))/h, and collapses to k-1 at flux touching.
These theorems do not integrate T-h along an orbit, first-hit, G1, G4,
or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16OrbitTh

/-- Cleared split of actual `T_h = q/h+k` versus the comparison. -/
theorem th_split_identity (q h k C ε T : ℝ) :
    (q + k * h) - (h + C * ε * T + C * ε ^ 3)
      - ((k - 1) * h + (q - C * ε * (T + ε ^ 2))) = 0 := by
  ring

/-- At `q = C ε (T + ε²)` the gap equals `k-1`, cleared of `h`. -/
theorem th_touching_identity (C ε T h k : ℝ) :
    ((C * ε * (T + ε ^ 2)) + k * h)
      - (h + C * ε * T + C * ε ^ 3)
      - (k - 1) * h = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16OrbitTh
