/-
Exact identities for the scale-dichotomy / next-atlas increment.

These theorems do not prove a physical C2 remainder, G1, or Hilbert XVI.
They do not apply Huzak–Kristiansen Theorem 2.4.
-/

import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ScaleDichotomy

/-- Blow-up exit height h = ε³ σ² η. -/
theorem blowup_height (ε σ η : ℝ) :
    ε ^ 3 * σ ^ 2 * η = ε ^ 3 * σ ^ 2 * η := rfl

/-- At equal ε and η the height ratio is the square of the scale ratio. -/
theorem blowup_height_ratio
    {ε σ1 σ2 η h1 h2 : ℝ}
    (hh1 : h1 = ε ^ 3 * σ1 ^ 2 * η)
    (hh2 : h2 = ε ^ 3 * σ2 ^ 2 * η)
    (hε : ε ≠ 0) (hη : η ≠ 0) (hσ2 : σ2 ≠ 0) :
    h1 / h2 = (σ1 / σ2) ^ 2 := by
  rw [hh1, hh2]
  field_simp [hε, hη, hσ2]

/-- Fold scale σ² = ε makes the exit height ε⁴. -/
theorem fold_exit_height {ε σ : ℝ} (hσ : σ ^ 2 = ε) :
    ε ^ 3 * σ ^ 2 = ε ^ 4 := by
  rw [hσ]
  ring

/-- Leading blow-up event exponent, with ℓ standing for log σ. -/
noncomputable def event_exponent (σ κ X ε ℓ : ℝ) : ℝ :=
  σ / X * κ + (2 * ε * σ / X) * ℓ

/-- The leading exponent is affine in kappa. -/
theorem event_exponent_kappa_increment
    {σ κ X ε ℓ : ℝ} (hX : X ≠ 0) :
    event_exponent σ κ X ε ℓ - event_exponent σ 0 X ε ℓ = σ / X * κ := by
  unfold event_exponent
  field_simp [hX]
  ring

/-- Discrete second kappa difference of the leading exponent vanishes. -/
theorem event_exponent_second_difference (σ X ε ℓ κ₀ h : ℝ) :
    (event_exponent σ (κ₀ + h) X ε ℓ - event_exponent σ κ₀ X ε ℓ)
      - (event_exponent σ κ₀ X ε ℓ - event_exponent σ (κ₀ - h) X ε ℓ) = 0 := by
  unfold event_exponent
  ring

/-- First kappa derivative of the leading exponent is σ / X. -/
theorem hasDerivAt_event_exponent (σ X ε ℓ κ : ℝ) :
    HasDerivAt (fun k : ℝ => event_exponent σ k X ε ℓ) (σ / X) κ := by
  unfold event_exponent
  have hmul : HasDerivAt (fun k : ℝ => (σ / X) * k) ((σ / X) * 1) κ :=
    (hasDerivAt_id κ).const_mul (σ / X)
  have hmul' : HasDerivAt (fun k : ℝ => (σ / X) * k) (σ / X) κ :=
    hmul.congr_deriv (by ring)
  exact hmul'.add_const ((2 * ε * σ / X) * ℓ)

/-- Exact joint-axis relation at negative lambda1. -/
theorem sep_plus_two_r1
    {lam1 sep r1 : ℝ} (hr1 : r1 = (-lam1 - sep) / 2) :
    2 * r1 + sep + lam1 = 0 := by
  rw [hr1]
  ring

/-- At fixed negative lambda1 the two smallness conditions cannot hold together. -/
theorem joint_sep_r1_exclusion
    {ℓ sep r1 : ℝ} (hsum : 2 * r1 + sep = ℓ)
    (hsep : sep < ℓ / 2) (hr1 : r1 < ℓ / 4) : False := by
  have : 2 * r1 + sep < ℓ := by
    have : 2 * r1 < ℓ / 2 := by
      have h2 : (0 : ℝ) < 2 := by norm_num
      have := mul_lt_mul_of_pos_left hr1 h2
      linarith
    linarith
  linarith

/-- The logarithmic inner coordinate on the named kill sequence is 1/ε. -/
theorem log_inner_on_kill {ε : ℝ} (hε : ε ≠ 0) :
    ε * Real.log (Real.exp (1 / ε ^ 2)) = 1 / ε := by
  rw [Real.log_exp]
  field_simp [hε]

end OmnibiasAnalytic.Dynamics.Hilbert16ScaleDichotomy
