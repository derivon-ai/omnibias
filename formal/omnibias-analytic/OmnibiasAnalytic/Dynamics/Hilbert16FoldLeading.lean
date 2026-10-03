/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Exact fold (sep = 0) I-map first derivative and leading C2 of log
sensitivity. Gronwall exp(C σ κ) is not this leading derivative.

These theorems do not prove a physical remainder, first-hit completeness,
chart O, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16FoldLeading

/-- Implicit I-map first derivative: `dx/dκ = δ² / (r - δ)`. -/
theorem fold_dx_identity
    {δ r : ℝ} (h : r - δ ≠ 0) :
    (δ ^ 2 / (r - δ)) * (r - δ) = δ ^ 2 := by
  field_simp [h]

/-- Second κ-derivative of the same implicit I-map. -/
theorem fold_d2x_identity
    {δ r : ℝ} (h : r - δ ≠ 0) :
    (-(δ ^ 3 * (2 * r - δ)) / (r - δ) ^ 3) * (r - δ) ^ 3
      + δ ^ 3 * (2 * r - δ) = 0 := by
  field_simp [h]
  ring

/-- First κ-derivative of `log(dx/dκ)` along the I-map. -/
theorem fold_log_d1_identity
    {δ r : ℝ} (h : r - δ ≠ 0) :
    (-(δ * (2 * r - δ)) / (r - δ) ^ 2) * (r - δ) ^ 2
      + δ * (2 * r - δ) = 0 := by
  field_simp [h]
  ring

/-- Leading C2 of `log(dx/dκ)` along the I-map. -/
theorem fold_log_c2_identity
    {δ r : ℝ} (h : r - δ ≠ 0) :
    (2 * r ^ 2 * δ ^ 2 / (r - δ) ^ 4) * (r - δ) ^ 4
      = 2 * r ^ 2 * δ ^ 2 := by
  field_simp [h]

/-- Reciprocal gap `δ = r / κ` recovers `dx/dκ = r / (κ² - κ)`. -/
theorem fold_reciprocal_gap
    {r κ : ℝ} (hr : r ≠ 0) (hκ : κ ≠ 0) (hκ1 : κ ≠ 1) :
    (r / κ) ^ 2 / (r - r / κ) = r / (κ ^ 2 - κ) := by
  have h1 : κ - 1 ≠ 0 := sub_ne_zero.mpr hκ1
  field_simp [hr, hκ, h1]
  ring

/-- Second κ-difference of a linear-in-κ leading `log D'` vanishes. -/
theorem chi_log_second_difference
    {r1 : ℝ} (hr1 : r1 ≠ 0) (c sep κ h : ℝ) :
    ((-c * (sep / r1) * (κ + h)) - (-c * (sep / r1) * κ))
      - ((-c * (sep / r1) * κ) - (-c * (sep / r1) * (κ - h))) = 0 := by
  field_simp [hr1]
  ring

/-- At the inner/outer interface `δ² = M ε`, relative remainder is `ρ / M`. -/
theorem interface_relative
    {ε ρ M : ℝ} (hε : ε ≠ 0) (hM : M ≠ 0) :
    (ε * ρ) / (M * ε) = ρ / M := by
  field_simp [hε, hM]

/-- Relative remainder identity `(B_eps - B)/B = ε ρ / δ²` in numerator form. -/
theorem relative_remainder_identity
    {δ ε ρ : ℝ} (hδ : δ ≠ 0) :
    (ε * ρ / δ ^ 2) * δ ^ 2 = ε * ρ := by
  field_simp [hδ]

/-- Actual `B_ε - B_-` from `ζ = -1 + β V + ε V Z` at `V = -ε x`. -/
theorem zeta_rho_identity (x L λ ε β Z : ℝ) :
    (L + λ * x - x ^ 2 * (-1 + β * (-ε * x) + ε * (-ε * x) * Z))
      - (L + λ * x + x ^ 2)
      = β * ε * x ^ 3 + ε ^ 2 * x ^ 3 * Z := by
  ring

/-- Lifted inner I-map: `dx/dκ = ((x-r)² + μ) / x`. -/
theorem lifted_dx_identity
    {x r μ : ℝ} (hx : x ≠ 0) :
    (((x - r) ^ 2 + μ) / x) * x = (x - r) ^ 2 + μ := by
  field_simp [hx]

/-- First κ-derivative of `log(dx/dκ)` on the lifted I-map. -/
theorem lifted_log_d1_identity
    {x r μ : ℝ} (hx : x ≠ 0) :
    ((2 * (x - r) * x - ((x - r) ^ 2 + μ)) / x ^ 2) * x ^ 2
      = 2 * (x - r) * x - ((x - r) ^ 2 + μ) := by
  field_simp [hx]

/-- On the lift, the Z remainder is relatively `O(ε)`. -/
theorem z_relative_on_lift
    {ε β Z : ℝ} (hε : ε ≠ 0) (hβ : β ≠ 0) :
    (ε ^ 2 * Z) / (β * ε) = (ε * Z) / β := by
  field_simp [hε, hβ]
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16FoldLeading
