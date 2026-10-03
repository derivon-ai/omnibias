/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Cancelled-N identities: the cubic slow-line remainder factors through
the v0 quadratic, the lambda source is O(ν²), and the L source is
O(ν⁴). These theorems do not enclose Z, bound T-h along the orbit,
first-hit, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16CancelledN

/-- `ℓ(v-v₀)+V-ν(v-v₀)²` versus the slow-line quadratic `ν v₀²+v₀-1`. -/
theorem ell_V_nu_identity (ν v v0 : ℝ) :
    (1 + 2 * ν * v) * (v - v0) + (1 - v - ν * v ^ 2)
      - ν * (v - v0) ^ 2 + (ν * v0 ^ 2 + v0 - 1) = 0 := by
  ring

/-- Unfolding slow line versus `slow0 + f0 + f1(v-v₀)`. -/
theorem delta_slow_identity (v v0 f0 f1 : ℝ) :
    (f0 - (3 * v0 ^ 2 + f1) * v0 + v0 ^ 3 + (3 * v0 ^ 2 + f1) * v - v ^ 3)
      - (-2 * v0 ^ 3 + 3 * v0 ^ 2 * v - v ^ 3) - f0 - f1 * (v - v0) = 0 := by
  ring

/-- Lambda source after clearing `k`: the O(ν) terms cancel on the root. -/
theorem t_lambda_cleared_identity (ν v v0 k λ1 : ℝ) :
    (1 + 2 * ν * v) * (-ν * k ^ 2 * λ1) * (v - v0)
      - ν * k ^ 2 * λ1 * (1 - v - ν * v ^ 2)
      + ν ^ 2 * k ^ 2 * λ1 * (v - v0) ^ 2
      + ν * k ^ 2 * λ1 * (ν * v0 ^ 2 + v0 - 1) = 0 := by
  ring

/-- `L` source after clearing `k ℓ₀²`: the O(ν²) terms cancel. -/
theorem t_L_cleared_identity (ν v v0 k L : ℝ) :
    (1 + 2 * ν * v) * (1 + 2 * ν * v0) * ν ^ 2 * k ^ 2 * L
      - 2 * ν ^ 3 * k ^ 2 * L * (1 + 2 * ν * v) * (v - v0)
      - ν ^ 2 * k ^ 2 * L * (1 + 2 * ν * v0) ^ 2
      + 4 * ν ^ 4 * k ^ 2 * L * (v - v0) ^ 2 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16CancelledN
