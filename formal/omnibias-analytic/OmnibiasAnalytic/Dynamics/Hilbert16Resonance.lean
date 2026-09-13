/-
Abstract resonance zero-count implications.

The displacement derivative is a positive factor times `1 - exp G`.
Strict convexity of the actual fixed-parameter function G excludes three
derivative zeros, hence excludes four displacement zeros by Rolle.
The module does not prove physical curvature/error estimates, control a
matched-parameter graph, or discharge any Hilbert XVI analytic hypotheses.
-/

import OmnibiasAnalytic.Dynamics.Hilbert16Rolle
import Mathlib.Analysis.Convex.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Tactic.Positivity

namespace OmnibiasAnalytic.Dynamics.Hilbert16Resonance

open Set Hilbert16Rolle

/-- Three ordered zeros contradict strict convexity. -/
theorem no_three_ordered_zeros_of_strictConvexOn
    {G : ℝ → ℝ} {s : Set ℝ} (hG : StrictConvexOn ℝ s G)
    {x y z : ℝ} (hx : x ∈ s) (hz : z ∈ s) (hxy : x < y) (hyz : y < z)
    (hGx : G x = 0) (hGy : G y = 0) (hGz : G z = 0) : False := by
  have hseg : y ∈ openSegment ℝ x z := by
    rw [openSegment_eq_Ioo (lt_trans hxy hyz)]
    exact ⟨hxy, hyz⟩
  have hlt := hG.lt_on_openSegment hx hz (ne_of_lt (lt_trans hxy hyz)) hseg
  simp only [hGx, hGy, hGz, max_self, lt_self_iff_false] at hlt

/-- Positive prefactors preserve the zeros of the logarithmic slope gap. -/
theorem normalized_slope_eq_zero_iff {p g : ℝ} (hp : 0 < p) :
    p * (1 - Real.exp g) = 0 ↔ g = 0 := by
  rw [mul_eq_zero]
  constructor
  · intro h
    rcases h with h | h
    · exact (ne_of_gt hp h).elim
    · apply Real.exp_injective
      rw [Real.exp_zero]
      linarith
  · intro h
    right
    simp [h]

/-- A strictly convex fixed-parameter logarithmic slope gap implies at most
three ordered displacement zeros. The derivative identity is required on
the entire interior interval, not merely at displacement zeros. -/
theorem no_four_ordered_displacement_zeros
    {F G p : ℝ → ℝ} {a b : ℝ}
    (hcont : ContinuousOn F (Icc a b))
    (hderiv : ∀ x ∈ Ioo a b, HasDerivAt F (p x * (1 - Real.exp (G x))) x)
    (hp : ∀ x ∈ Ioo a b, 0 < p x)
    (hG : StrictConvexOn ℝ (Icc a b) G)
    {x₁ x₂ x₃ x₄ : ℝ} (ha : a ≤ x₁) (h₁₂ : x₁ < x₂)
    (h₂₃ : x₂ < x₃) (h₃₄ : x₃ < x₄) (hb : x₄ ≤ b)
    (hF₁ : F x₁ = 0) (hF₂ : F x₂ = 0) (hF₃ : F x₃ = 0)
    (hF₄ : F x₄ = 0) : False := by
  obtain ⟨c₁, hc₁, hz₁⟩ := exists_derivative_zero_between hcont hderiv
    ha h₁₂ (by linarith) hF₁ hF₂
  obtain ⟨c₂, hc₂, hz₂⟩ := exists_derivative_zero_between hcont hderiv
    (by linarith) h₂₃ (by linarith) hF₂ hF₃
  obtain ⟨c₃, hc₃, hz₃⟩ := exists_derivative_zero_between hcont hderiv
    (by linarith) h₃₄ hb hF₃ hF₄
  have hc₁dom : c₁ ∈ Ioo a b := ⟨by linarith [hc₁.1], by linarith [hc₁.2]⟩
  have hc₂dom : c₂ ∈ Ioo a b := ⟨by linarith [hc₂.1], by linarith [hc₂.2]⟩
  have hc₃dom : c₃ ∈ Ioo a b := ⟨by linarith [hc₃.1], by linarith [hc₃.2]⟩
  apply no_three_ordered_zeros_of_strictConvexOn hG (x := c₁) (y := c₂) (z := c₃)
    ⟨hc₁dom.1.le, hc₁dom.2.le⟩ ⟨hc₃dom.1.le, hc₃dom.2.le⟩
    (by linarith [hc₁.2, hc₂.1]) (by linarith [hc₂.2, hc₃.1])
  · exact (normalized_slope_eq_zero_iff (hp c₁ hc₁dom)).mp hz₁
  · exact (normalized_slope_eq_zero_iff (hp c₂ hc₂dom)).mp hz₂
  · exact (normalized_slope_eq_zero_iff (hp c₃ hc₃dom)).mp hz₃

/-- Positive actual second derivative supplies the strict-convexity premise. -/
theorem no_four_ordered_displacement_zeros_of_positive_curvature
    {F G p : ℝ → ℝ} {a b : ℝ}
    (hcont : ContinuousOn F (Icc a b))
    (hderiv : ∀ x ∈ Ioo a b, HasDerivAt F (p x * (1 - Real.exp (G x))) x)
    (hp : ∀ x ∈ Ioo a b, 0 < p x)
    (hGcont : ContinuousOn G (Icc a b))
    (hcurvature : ∀ x ∈ Ioo a b, 0 < (deriv^[2] G) x)
    {x₁ x₂ x₃ x₄ : ℝ} (ha : a ≤ x₁) (h₁₂ : x₁ < x₂)
    (h₂₃ : x₂ < x₃) (h₃₄ : x₃ < x₄) (hb : x₄ ≤ b)
    (hF₁ : F x₁ = 0) (hF₂ : F x₂ = 0) (hF₃ : F x₃ = 0)
    (hF₄ : F x₄ = 0) : False := by
  apply no_four_ordered_displacement_zeros hcont hderiv hp
    (strictConvexOn_of_deriv2_pos (convex_Icc a b) hGcont ?_) ha h₁₂ h₂₃ h₃₄ hb
    hF₁ hF₂ hF₃ hF₄
  simpa only [interior_Icc] using hcurvature

/-- A relative error strictly below the positive leading budget leaves
strict positive curvature. This does not establish an actual error bound. -/
theorem positive_curvature_budget {a b omega u eta remainder : ℝ}
    (ha : 0 < a) (hb : 0 < b) (ho : 0 < omega) (hu : 0 < u)
    (heta : eta < 1) (herror : |remainder| ≤ eta * (a * omega + b * u)) :
    0 < a * omega + b * u + remainder := by
  have hbudget : 0 < a * omega + b * u := by positivity
  have hlower := neg_abs_le remainder
  nlinarith

/-- A single increasing core, with downward crossings at all zeros outside
it, excludes four zeros on the whole interval. The core may be empty or
a singleton; no derivative-sign condition is imposed at nonzeros outside. -/
theorem no_four_zeros_with_increasing_core
    {F F' : ℝ → ℝ} {a b l r : ℝ}
    (hcont : ContinuousOn F (Icc a b))
    (hderiv : ∀ x ∈ Icc a b, F x = 0 → HasDerivAt F (F' x) x)
    (hneg : ∀ x ∈ Icc a b, x ∉ Icc l r → F x = 0 → F' x < 0)
    (hcore : StrictMonoOn F (Icc l r))
    {x₁ x₂ x₃ x₄ : ℝ} (ha : a ≤ x₁) (h₁₂ : x₁ < x₂)
    (h₂₃ : x₂ < x₃) (h₃₄ : x₃ < x₄) (hb : x₄ ≤ b)
    (hF₁ : F x₁ = 0) (hF₂ : F x₂ = 0) (hF₃ : F x₃ = 0)
    (hF₄ : F x₄ = 0) : False := by
  by_cases hleft : x₂ < l
  · have hsub : Icc x₁ x₂ ⊆ Icc a b := by
      intro t ht
      exact ⟨le_trans ha ht.1, by linarith [ht.2]⟩
    apply no_two_ordered_zeros_of_negative_derivative_at_zeros (hcont.mono hsub)
      (fun t ht => hderiv t (hsub ht)) ?_ le_rfl h₁₂ le_rfl hF₁ hF₂
    intro t ht hFt
    apply hneg t (hsub ht) ?_ hFt
    intro hmem
    linarith [ht.2, hmem.1]
  by_cases hright : r < x₃
  · have hsub : Icc x₃ x₄ ⊆ Icc a b := by
      intro t ht
      exact ⟨by linarith [ht.1], le_trans ht.2 hb⟩
    apply no_two_ordered_zeros_of_negative_derivative_at_zeros (hcont.mono hsub)
      (fun t ht => hderiv t (hsub ht)) ?_ le_rfl h₃₄ le_rfl hF₃ hF₄
    intro t ht hFt
    apply hneg t (hsub ht) ?_ hFt
    intro hmem
    linarith [ht.1, hmem.2]
  have hx₂ : x₂ ∈ Icc l r := ⟨by linarith, by linarith⟩
  have hx₃ : x₃ ∈ Icc l r := ⟨by linarith, by linarith⟩
  have hlt := hcore hx₂ hx₃ h₂₃
  simp only [hF₂, hF₃, lt_self_iff_false] at hlt

/-- A strictly convex logarithmic gap with nonpositive endpoint values
produces an increasing displacement core, including a singleton core. -/
theorem strictMonoOn_of_nonpositive_convex_gap_core
    {F G p : ℝ → ℝ} {l r : ℝ}
    (hcont : ContinuousOn F (Icc l r))
    (hderiv : ∀ x ∈ Ioo l r, HasDerivAt F (p x * (1 - Real.exp (G x))) x)
    (hp : ∀ x ∈ Ioo l r, 0 < p x)
    (hG : StrictConvexOn ℝ (Icc l r) G)
    (hGl : G l ≤ 0) (hGr : G r ≤ 0) : StrictMonoOn F (Icc l r) := by
  apply strictMonoOn_of_deriv_pos (convex_Icc l r) hcont
  intro x hx
  have hxI : x ∈ Ioo l r := by simpa only [interior_Icc] using hx
  have hlr : l < r := lt_trans hxI.1 hxI.2
  have hseg : x ∈ openSegment ℝ l r := by
    rw [openSegment_eq_Ioo hlr]
    exact hxI
  have hGx : G x < 0 := lt_of_lt_of_le
    (hG.lt_on_openSegment ⟨le_rfl, hlr.le⟩ ⟨hlr.le, le_rfl⟩ hlr.ne hseg)
    (max_le hGl hGr)
  rw [(hderiv x hxI).deriv]
  apply mul_pos (hp x hxI)
  have hexp : Real.exp (G x) < 1 := by
    simpa only [Real.exp_zero] using Real.exp_lt_exp.mpr hGx
  linarith

end OmnibiasAnalytic.Dynamics.Hilbert16Resonance
