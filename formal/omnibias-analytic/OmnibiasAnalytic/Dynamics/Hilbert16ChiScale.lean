/-
Linear-saddle χ-scale identities and the frozen-exponent obstruction.

These theorems concern the exact model X' = -ε s X, Y' = r Y and the
matching coordinate χ = (s/r) κ. The Python evaluator, any physical
passage remainder, and any Dulac membership statement are outside these
results. No Hilbert XVI or graphic-cyclicity theorem is asserted.
-/

import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ChiScale

/-- The linear-saddle matching coordinate. -/
noncomputable def chi (s r κ : ℝ) : ℝ := s * κ / r

/-- The exact linear exit is the exponential of minus χ. -/
theorem linear_exit_eq_exp_neg_chi {s r κ : ℝ} (hr : r ≠ 0) :
    Real.exp ((-s / r) * κ) = Real.exp (-chi s r κ) := by
  unfold chi
  congr 1
  field_simp [hr]

/-- Uniform χ-derivative of the linear exit. -/
theorem hasDerivAt_exp_neg (χ : ℝ) :
    HasDerivAt (fun t : ℝ => Real.exp (-t)) (-Real.exp (-χ)) χ := by
  have h : HasDerivAt (fun t : ℝ => Real.exp (-t))
      (Real.exp (-χ) * -1) χ :=
    (Real.hasDerivAt_exp (-χ)).comp χ (hasDerivAt_id χ).neg
  exact h.congr_deriv (by ring)

/-- Kappa derivative of the same closed form. -/
theorem hasDerivAt_linear_exit_kappa (s r κ : ℝ) :
    HasDerivAt (fun k : ℝ => Real.exp ((-s / r) * k))
      ((-s / r) * Real.exp ((-s / r) * κ)) κ := by
  have hmul : HasDerivAt (fun k : ℝ => (-s / r) * k) ((-s / r) * 1) κ :=
    (hasDerivAt_id κ).const_mul (-s / r)
  have hmul' : HasDerivAt (fun k : ℝ => (-s / r) * k) (-s / r) κ :=
    hmul.congr_deriv (by ring)
  have h : HasDerivAt (fun k : ℝ => Real.exp ((-s / r) * k))
      (Real.exp ((-s / r) * κ) * (-s / r)) κ :=
    (Real.hasDerivAt_exp ((-s / r) * κ)).comp κ hmul'
  exact h.congr_deriv (by ring)

/-- Fixed χ with vanishing separation sends kappa to infinity. -/
theorem chi_of_scaled_kappa {s r χ : ℝ} (hs : s ≠ 0) (hr : r ≠ 0) :
    chi s r (χ * r / s) = χ := by
  unfold chi
  field_simp

/-- A pre-rectangle time of order 1/s produces an O(1) χ-threshold. -/
theorem chi_of_reciprocal_sep {s r K : ℝ} (hs : s ≠ 0) (hr : r ≠ 0) :
    chi s r (K / s) = K / r := by
  unfold chi
  field_simp

/-- Sensitivity ratio of the linear exit against a frozen exponential. -/
theorem kappa_sensitivity_ratio
    {s r κ C γ : ℝ} (hC : 0 < C) (hr : 0 < r) :
    (s / r * Real.exp ((-s / r) * κ)) / (C * Real.exp (-γ * κ)) =
      s / (r * C) * Real.exp ((γ - s / r) * κ) := by
  have hC0 : C ≠ 0 := ne_of_gt hC
  have hr0 : r ≠ 0 := ne_of_gt hr
  calc
    (s / r * Real.exp ((-s / r) * κ)) / (C * Real.exp (-γ * κ)) =
        (s / r / C) * (Real.exp ((-s / r) * κ) / Real.exp (-γ * κ)) := by
      field_simp
    _ = (s / (r * C)) * Real.exp ((-s / r) * κ - (-γ * κ)) := by
      rw [Real.exp_sub]
      field_simp
    _ = s / (r * C) * Real.exp ((γ - s / r) * κ) := by
      apply congrArg (fun t => s / (r * C) * Real.exp t)
      ring

/-- No frozen (C, γ) dominates the kappa sensitivity for every small s. -/
theorem frozen_exponent_obstruction
    {C γ r s0 : ℝ} (hC : 0 < C) (hγ : 0 < γ) (hr : 0 < r) (hs0 : 0 < s0) :
    ∃ s κ : ℝ, 0 < s ∧ s ≤ s0 ∧ 0 < κ ∧
      C * Real.exp (-γ * κ) < s / r * Real.exp ((-s / r) * κ) := by
  set s := min s0 (r * γ / 2)
  have hs_pos : 0 < s := (lt_min_iff (a := (0 : ℝ))).2 ⟨hs0, by positivity⟩
  have hs_le : s ≤ s0 := min_le_left _ _
  have hα : 0 < γ - s / r := by
    have hsr : s / r ≤ γ / 2 := (div_le_iff₀ hr).2 (by
      have : s ≤ r * γ / 2 := min_le_right _ _
      linarith)
    linarith
  set α := γ - s / r
  have hα_pos : 0 < α := hα
  set κ := (r * C / s + 1) / α
  have hκ : 0 < κ := div_pos (by positivity) hα_pos
  refine ⟨s, κ, hs_pos, hs_le, hκ, ?_⟩
  have hακ : α * κ = r * C / s + 1 := by
    have hα0 : α ≠ 0 := ne_of_gt hα_pos
    calc
      α * κ = α * ((r * C / s + 1) / α) := rfl
      _ = r * C / s + 1 := by field_simp [hα0]
  have hexp_lb : r * C / s < Real.exp (α * κ) := by
    have hle : α * κ + 1 ≤ Real.exp (α * κ) := Real.add_one_le_exp (α * κ)
    have hsum : α * κ + 1 = r * C / s + 2 := by
      rw [hακ]
      ring
    rw [hsum] at hle
    have : r * C / s < r * C / s + 2 := by linarith
    linarith
  have hgt : 1 < s / (r * C) * Real.exp (α * κ) := by
    have hrC : 0 < r * C := by positivity
    have hs0' : s ≠ 0 := ne_of_gt hs_pos
    have hrC0 : r * C ≠ 0 := ne_of_gt hrC
    have hone : s / (r * C) * (r * C / s) = 1 := by field_simp [hs0', hrC0]
    have hmul :=
      mul_lt_mul_of_pos_left hexp_lb (show 0 < s / (r * C) by positivity)
    calc
      (1 : ℝ) = s / (r * C) * (r * C / s) := hone.symm
      _ < s / (r * C) * Real.exp (α * κ) := hmul
  have hratio := kappa_sensitivity_ratio (s := s) (κ := κ) (γ := γ) hC hr
  have hcmp :
      s / r * Real.exp ((-s / r) * κ) / (C * Real.exp (-γ * κ)) =
        s / (r * C) * Real.exp (α * κ) := by
    simpa [α] using hratio
  have hden : 0 < C * Real.exp (-γ * κ) := by positivity
  rw [← hcmp] at hgt
  exact (one_lt_div hden).mp hgt

end OmnibiasAnalytic.Dynamics.Hilbert16ChiScale
