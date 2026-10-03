/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
k = 1 + O(ε) on the C = 0 normal chart, and the cubic remainder
prefactor past the two-root leading term. These theorems do not bound
Z, T-h along the orbit, height-section first-hit, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16KZetaRemainder

/-- Unfolding turning-geometry `k - 1 = -A ν v + C ν² v²`. -/
theorem k_turn_identity (A ν v C : ℝ) :
    (1 - A * ν * v + C * ν ^ 2 * v ^ 2) - 1
      - (-A * ν * v + C * ν ^ 2 * v ^ 2) = 0 := by
  ring

/-- `C = 0` normal `k = ℓ (1 - A ν v)` versus the expanded polynomial. -/
theorem k_normal_identity (A ν v : ℝ) :
    (1 + 2 * ν * v) * (1 - A * ν * v)
      - (1 + (2 - A) * ν * v - 2 * A * ν ^ 2 * v ^ 2) = 0 := by
  ring

/-- First-order jet on `V = 1 - v - ν v²`. -/
theorem k_lead_embed_identity (ν v : ℝ) :
    (1 - ν * ((1 - v - ν * v ^ 2) - 1))
      - (1 + ν * v + ν ^ 2 * v ^ 2) = 0 := by
  ring

/-- Exact `O(ν²)` gap between normal `k` and the V-jet at `A = 1`. -/
theorem k_normal_vs_lead_identity (ν v : ℝ) :
    (1 + 2 * ν * v) * (1 - ν * v)
      - (1 - ν * ((1 - v - ν * v ^ 2) - 1))
      + 3 * ν ^ 2 * v ^ 2 = 0 := by
  ring

/-- Box gap `9 - (V-1)²` versus `8 + 2V - V²`. -/
theorem k_box_gap_identity (V : ℝ) :
    (9 - (V - 1) ^ 2) - (8 + 2 * V - V ^ 2) = 0 := by
  ring

/-- Cubic `q` versus two-root leading plus `ε⁴ x³ / 3`. -/
theorem cubic_vs_tworoot_identity (ε x r1 r2 : ℝ) :
    (r1 * r2) * ε ^ 3 + (-(r1 + r2)) * ε ^ 2 * (ε * x)
      + ε * (ε * x) ^ 2 * (1 + (ε * x) / 3)
      - ε ^ 3 * (x - r1) * (x - r2) - ε ^ 4 * x ^ 3 / 3 = 0 := by
  ring

/-- Relative cubic prefactor `2(ε² + T) - w² = 2 ε²` at `T = w²/2`. -/
theorem relative_prefactor_identity (ε w : ℝ) :
    2 * (ε ^ 2 + w ^ 2 / 2) - w ^ 2 - 2 * ε ^ 2 = 0 := by
  ring

/-- `|V-1| ≤ 3` on `|V| ≤ 2`, so `|k_lead-1| ≤ 3 ν` for `ν ≥ 0`. -/
theorem k_box_nonneg {V : ℝ} (hlo : -2 ≤ V) (hhi : V ≤ 2) :
    0 ≤ 9 - (V - 1) ^ 2 := by
  have : (V - 1) ^ 2 ≤ 9 := by nlinarith
  nlinarith

end OmnibiasAnalytic.Dynamics.Hilbert16KZetaRemainder
