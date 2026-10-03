/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Holomorphic Z_v identities on the lambda=0 slice: termwise q1_v matches
the closed form, the product-rule numerator for
d/dv[(wall+ell0)/wall^3] matches nu (-2 wall - 3 ell0), and the
quotient-rule remainder for Z0 clears after multiplying by the
denominator. These theorems do not enclose fold Z_x, sep>0, first-hit,
G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ZVBound

/-- Termwise `q1_v` minus the closed form. -/
theorem q1_v_identity (nu v v0 : ℝ) :
    ((-3) * v0 + nu * ((-3) * v0 * 2 * (v0 + v))
        - (nu ^ 2) * v0 * (3 * (v0 + v) ^ 2) + (2 + nu * v0))
      - ((-3) * v0 - 6 * nu * v0 * (v0 + v)
        - 3 * (nu ^ 2) * v0 * ((v0 + v) ^ 2) + 2 + nu * v0) = 0 := by
  ring

/-- Product-rule numerator versus `nu (-2 wall - 3 ell0)`. -/
theorem wall_ratio_v_identity (nu v v0 : ℝ) :
    (nu * (1 + nu * (v0 + v))
        - 3 * nu * ((1 + nu * (v0 + v)) + (1 + 2 * nu * v0)))
      - (nu * ((-2) * (1 + nu * (v0 + v)) - 3 * (1 + 2 * nu * v0))) = 0 := by
  ring

/-- Quotient-rule remainder after cancelling `g` and the `wall^4` pole. -/
theorem zv_quotient_identity (nu v v0 q1 q1v : ℝ) :
    ((1 + 2 * nu * v0) * (q1v * (1 + nu * (v0 + v)) - 3 * nu * q1)
        * (9 * v0 * v0 * (1 + nu * (v0 + v)) ^ 3))
      - (((1 + 2 * nu * v0) * q1v) * (9 * v0 * v0 * (1 + nu * (v0 + v)) ^ 3)
        - ((1 + 2 * nu * v0) * q1)
          * (27 * v0 * v0 * (1 + nu * (v0 + v)) ^ 2 * nu))
        * (1 + nu * (v0 + v)) = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ZVBound
