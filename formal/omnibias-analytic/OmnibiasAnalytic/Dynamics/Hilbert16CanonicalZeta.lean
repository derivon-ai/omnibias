/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Actual r = -1 slow-line zeta identities on the lambda0 = lambda1 = 0
slice. These theorems do not bound the fold (L, λ₁) remainder, first-hit,
G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16CanonicalZeta

/-- Factorization `V = (v0 - v)(1 + ν(v0 + v))` once `v0` is a slow-line root. -/
theorem V_factor_identity
    {ν v v0 : ℝ}
    (h : ν * v0 ^ 2 + v0 - 1 = 0) :
    (1 - v - ν * v ^ 2) - (v0 - v) * (1 + ν * (v0 + v)) = 0 := by
  have h1 :
      (1 - v - ν * v ^ 2) - (v0 - v) * (1 + ν * (v0 + v))
        = -(ν * v0 ^ 2 + v0 - 1) := by
    ring
  rw [h1, h]
  ring

/-- `k l = 3 v0` on the `lambda = 0` slice. -/
theorem k_lambda0_identity
    {ν v0 : ℝ} (hl : 1 + 2 * ν * v0 ≠ 0) :
    (3 * v0 / (1 + 2 * ν * v0)) * (1 + 2 * ν * v0) = 3 * v0 := by
  field_simp [hl]

/-- Closed-form `zeta` at `v = v0` is `-1`. -/
theorem zeta_at_zero_identity
    {ν v0 : ℝ}
    (hv0 : v0 ≠ 0)
    (hl : 1 + 2 * ν * v0 ≠ 0) :
    -((1 + 2 * ν * v0) * (1 + 2 * ν * v0) * (v0 + 2 * v0))
        / (3 * v0 * (1 + ν * (v0 + v0)) ^ 2)
      = -1 := by
  have hD : 1 + ν * (v0 + v0) ≠ 0 := by
    have : 1 + ν * (v0 + v0) = 1 + 2 * ν * v0 := by ring
    simpa [this] using hl
  field_simp [hv0, hl, hD]
  ring

/-- Limiting cubic: `zeta(V, 0) = -1 + V/3` at `ν = 0`, `v = 1 - V`. -/
theorem limiting_cubic_identity (V : ℝ) :
    -((1 : ℝ) * 1 * ((1 - V) + 2 * 1)) / (3 * 1 * 1 ^ 2)
      = -1 + V / 3 := by
  ring

/-- Linear jet identity `1/(3 v0 l) = 1/(k l²)`. -/
theorem beta_linear_jet_identity
    {ν v0 : ℝ}
    (hv0 : v0 ≠ 0)
    (hl : 1 + 2 * ν * v0 ≠ 0) :
    (1 : ℝ) / (3 * v0 * (1 + 2 * ν * v0))
      = 1
          / ((3 * v0 / (1 + 2 * ν * v0))
            * (1 + 2 * ν * v0) ^ 2) := by
  field_simp [hv0, hl]
  ring

/-- Rearrangement of the `lambda0 = 0` implicit `k` equation. -/
theorem k_implicit_lambda0_clear
    {ν v0 k λ1 : ℝ} (hl : 1 + 2 * ν * v0 ≠ 0) :
    (k - 3 * v0 / (1 + 2 * ν * v0)
        - ν ^ 2 * k ^ 2 * λ1 / (1 + 2 * ν * v0) ^ 2)
      * (1 + 2 * ν * v0) ^ 2
      = k * (1 + 2 * ν * v0) ^ 2 - 3 * v0 * (1 + 2 * ν * v0)
        - ν ^ 2 * k ^ 2 * λ1 := by
  field_simp [hl]
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16CanonicalZeta
