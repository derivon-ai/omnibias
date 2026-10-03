/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Comparison-bootstrap integral of T-h: the slope q/h+k-1 splits after
T ≤ K(ε²+h) into a linear O(ε) piece plus an ε³/h piece. These theorems
do not enclose a Lohner-validated (V,h) orbit, first-hit, G1, G4, or
Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ThIntegral

/-- Cleared `(T-h)_h = q/h + k - 1`. -/
theorem th_dh_identity (q h k : ℝ) :
    (q + k * h - h) - (q + (k - 1) * h) = 0 := by
  ring

/-- Plug `T ≤ K(ε²+h)` into `C ε (ε²+T)`, cleared of the height denominator. -/
theorem majorant_split_identity (C K ε h : ℝ) :
    C * ε * (ε ^ 2 + K * (ε ^ 2 + h))
      - (C * K * ε * h + C * (1 + K) * ε ^ 3) = 0 := by
  ring

/-- Linear-slope orbit of `T_h = 1+α` increments `T-h` by `α Δh`. -/
theorem ftc_linear_identity (Te he α h : ℝ) :
    ((Te + (α + 1) * (h - he)) - h) - (Te - he) - α * (h - he) = 0 := by
  ring

/-- The `ε^{3/2}` prefactor is `ε³` times `s² = hmax/(ε³ y0)`. -/
theorem log_sqrt_prefactor_identity (ε s y0 hmax : ℝ) :
    ε ^ 6 * s ^ 2 * y0 - ε ^ 3 * hmax
      - ε ^ 3 * (ε ^ 3 * s ^ 2 * y0 - hmax) = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ThIntegral
