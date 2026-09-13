/-
Abstract zero-count inference for real return-map displacements.

Mathlib's Rolle theorem supplies a distinct derivative zero between every
two consecutive function zeros. The hypotheses below concern a continuous
function on a closed interval and its actual derivative on the interior.
No physical flow, passage estimate, parameter uniformity, or Hilbert XVI
claim is encoded by this module.
-/

import Mathlib.Analysis.Calculus.LocalExtr.Rolle
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Tactic.Linarith

namespace OmnibiasAnalytic.Dynamics.Hilbert16Rolle

open Set Filter Topology

/-- Any two zeros in the indicated set are the same point. -/
def AtMostOneZero (g : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x ∈ s, ∀ y ∈ s, g x = 0 → g y = 0 → x = y

/-- Any three zeros in the indicated set include an equal pair. -/
def AtMostTwoZeros (g : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x ∈ s, ∀ y ∈ s, ∀ z ∈ s,
    g x = 0 → g y = 0 → g z = 0 → x = y ∨ x = z ∨ y = z

variable {f f' : ℝ → ℝ} {a b : ℝ}

/-- Rolle on a subinterval, with an explicitly supplied actual derivative. -/
theorem exists_derivative_zero_between
    (hcont : ContinuousOn f (Icc a b))
    (hderiv : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x)
    {x y : ℝ} (hx : a ≤ x) (hxy : x < y) (hy : y ≤ b)
    (hfx : f x = 0) (hfy : f y = 0) :
    ∃ c ∈ Ioo x y, f' c = 0 := by
  apply exists_hasDerivAt_eq_zero hxy
  · exact hcont.mono (fun t ht => ⟨le_trans hx ht.1, le_trans ht.2 hy⟩)
  · exact hfx.trans hfy.symm
  · intro t ht
    exact hderiv t ⟨lt_of_le_of_lt hx ht.1, lt_of_lt_of_le ht.2 hy⟩

/-- At most one derivative zero excludes three ordered function zeros. -/
theorem no_three_ordered_zeros
    (hcont : ContinuousOn f (Icc a b))
    (hderiv : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x)
    (hone : AtMostOneZero f' (Ioo a b))
    {x₁ x₂ x₃ : ℝ} (ha : a ≤ x₁) (h₁₂ : x₁ < x₂)
    (h₂₃ : x₂ < x₃) (hb : x₃ ≤ b)
    (hf₁ : f x₁ = 0) (hf₂ : f x₂ = 0) (hf₃ : f x₃ = 0) : False := by
  obtain ⟨c₁, hc₁, hz₁⟩ := exists_derivative_zero_between hcont hderiv
    ha h₁₂ (le_trans h₂₃.le hb) hf₁ hf₂
  obtain ⟨c₂, hc₂, hz₂⟩ := exists_derivative_zero_between hcont hderiv
    (le_trans ha h₁₂.le) h₂₃ hb hf₂ hf₃
  have heq : c₁ = c₂ := hone c₁ ⟨by linarith [hc₁.1], by linarith [hc₁.2]⟩
    c₂ ⟨by linarith [hc₂.1], by linarith [hc₂.2]⟩ hz₁ hz₂
  linarith [hc₁.2, hc₂.1]

/-- At most two derivative zeros excludes four ordered function zeros. -/
theorem no_four_ordered_zeros
    (hcont : ContinuousOn f (Icc a b))
    (hderiv : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x)
    (htwo : AtMostTwoZeros f' (Ioo a b))
    {x₁ x₂ x₃ x₄ : ℝ} (ha : a ≤ x₁) (h₁₂ : x₁ < x₂)
    (h₂₃ : x₂ < x₃) (h₃₄ : x₃ < x₄) (hb : x₄ ≤ b)
    (hf₁ : f x₁ = 0) (hf₂ : f x₂ = 0) (hf₃ : f x₃ = 0)
    (hf₄ : f x₄ = 0) : False := by
  obtain ⟨c₁, hc₁, hz₁⟩ := exists_derivative_zero_between hcont hderiv
    ha h₁₂ (by linarith) hf₁ hf₂
  obtain ⟨c₂, hc₂, hz₂⟩ := exists_derivative_zero_between hcont hderiv
    (by linarith) h₂₃ (by linarith) hf₂ hf₃
  obtain ⟨c₃, hc₃, hz₃⟩ := exists_derivative_zero_between hcont hderiv
    (by linarith) h₃₄ hb hf₃ hf₄
  have hpairs := htwo c₁ ⟨by linarith [hc₁.1], by linarith [hc₁.2]⟩
    c₂ ⟨by linarith [hc₂.1], by linarith [hc₂.2]⟩
    c₃ ⟨by linarith [hc₃.1], by linarith [hc₃.2]⟩ hz₁ hz₂ hz₃
  rcases hpairs with h | h | h <;> linarith [hc₁.2, hc₂.1, hc₂.2, hc₃.1]

/-- The three-zero exclusion with the usual `deriv` operator. -/
theorem no_three_ordered_zeros_deriv
    (hcont : ContinuousOn f (Icc a b))
    (hdiff : DifferentiableOn ℝ f (Ioo a b))
    (hone : AtMostOneZero (deriv f) (Ioo a b))
    {x₁ x₂ x₃ : ℝ} (ha : a ≤ x₁) (h₁₂ : x₁ < x₂)
    (h₂₃ : x₂ < x₃) (hb : x₃ ≤ b)
    (hf₁ : f x₁ = 0) (hf₂ : f x₂ = 0) (hf₃ : f x₃ = 0) : False := by
  apply no_three_ordered_zeros hcont ?_ hone ha h₁₂ h₂₃ hb hf₁ hf₂ hf₃
  intro x hx
  exact (hdiff.differentiableAt (isOpen_Ioo.mem_nhds hx)).hasDerivAt

/-- The four-zero exclusion with the usual `deriv` operator. -/
theorem no_four_ordered_zeros_deriv
    (hcont : ContinuousOn f (Icc a b))
    (hdiff : DifferentiableOn ℝ f (Ioo a b))
    (htwo : AtMostTwoZeros (deriv f) (Ioo a b))
    {x₁ x₂ x₃ x₄ : ℝ} (ha : a ≤ x₁) (h₁₂ : x₁ < x₂)
    (h₂₃ : x₂ < x₃) (h₃₄ : x₃ < x₄) (hb : x₄ ≤ b)
    (hf₁ : f x₁ = 0) (hf₂ : f x₂ = 0) (hf₃ : f x₃ = 0)
    (hf₄ : f x₄ = 0) : False := by
  apply no_four_ordered_zeros hcont ?_ htwo ha h₁₂ h₂₃ h₃₄ hb hf₁ hf₂ hf₃ hf₄
  intro x hx
  exact (hdiff.differentiableAt (isOpen_Ioo.mem_nhds hx)).hasDerivAt

/-- A negative derivative at a zero gives negative values immediately to its right. -/
theorem exists_negative_right_of_negative_derivative
    {x y slopeValue : ℝ} (hxy : x < y)
    (hfx : f x = 0) (hd : HasDerivAt f slopeValue x) (hdneg : slopeValue < 0) :
    ∃ t ∈ Ioo x y, f t < 0 := by
  have hslope := (hasDerivAt_iff_tendsto_slope_left_right.mp hd).2
  have hneg : ∀ᶠ t in 𝓝[>] x, slope f x t < 0 :=
    hslope.eventually (Iio_mem_nhds hdneg)
  have hint : ∀ᶠ t in 𝓝[>] x, t ∈ Ioo x y := Ioo_mem_nhdsGT hxy
  obtain ⟨t, htI, htslope⟩ := (hint.and hneg).exists
  refine ⟨t, htI, ?_⟩
  rw [slope_def_field, hfx, sub_zero] at htslope
  rcases div_neg_iff.mp htslope with h | h <;> linarith [htI.1]

/-- A negative derivative at a zero gives positive values immediately to its left. -/
theorem exists_positive_left_of_negative_derivative
    {x y slopeValue : ℝ} (hxy : x < y)
    (hfy : f y = 0) (hd : HasDerivAt f slopeValue y) (hdneg : slopeValue < 0) :
    ∃ t ∈ Ioo x y, 0 < f t := by
  have hslope := (hasDerivAt_iff_tendsto_slope_left_right.mp hd).1
  have hneg : ∀ᶠ t in 𝓝[<] y, slope f y t < 0 :=
    hslope.eventually (Iio_mem_nhds hdneg)
  have hint : ∀ᶠ t in 𝓝[<] y, t ∈ Ioo x y := Ioo_mem_nhdsLT hxy
  obtain ⟨t, htI, htslope⟩ := (hint.and hneg).exists
  refine ⟨t, htI, ?_⟩
  rw [slope_def_field, hfy, sub_zero] at htslope
  rcases div_neg_iff.mp htslope with h | h <;> linarith [htI.2]

/-- Negative derivatives at every zero exclude two ordered zeros. Endpoint
zeros require actual derivatives as well. No differentiability or derivative
sign condition is required away from the zero set. -/
theorem no_two_ordered_zeros_of_negative_derivative_at_zeros
    (hcont : ContinuousOn f (Icc a b))
    (hderiv : ∀ x ∈ Icc a b, f x = 0 → HasDerivAt f (f' x) x)
    (hneg : ∀ x ∈ Icc a b, f x = 0 → f' x < 0)
    {x y : ℝ} (hx : a ≤ x) (hxy : x < y) (hy : y ≤ b)
    (hfx : f x = 0) (hfy : f y = 0) : False := by
  have hxmem : x ∈ Icc a b := ⟨hx, le_trans hxy.le hy⟩
  obtain ⟨r, hr, hfr⟩ := exists_negative_right_of_negative_derivative
    hxy hfx (hderiv x hxmem hfx) (hneg x hxmem hfx)
  have hrcont : ContinuousOn f (Icc r y) :=
    hcont.mono (fun t ht => ⟨by linarith [hr.1, ht.1], le_trans ht.2 hy⟩)
  let zeros : Set ℝ := Icc r y ∩ f ⁻¹' {0}
  have hclosed : IsClosed zeros :=
    hrcont.preimage_isClosed_of_isClosed isClosed_Icc isClosed_singleton
  have hcompact : IsCompact zeros :=
    isCompact_Icc.of_isClosed_subset hclosed (fun t ht => ht.1)
  have hnonempty : zeros.Nonempty := ⟨y, ⟨⟨hr.2.le, le_rfl⟩, hfy⟩⟩
  obtain ⟨c, hc⟩ := hcompact.exists_isLeast hnonempty
  have hcI : c ∈ Icc r y := hc.1.1
  have hfc : f c = 0 := hc.1.2
  have hrc : r < c := by
    have hne : r ≠ c := by intro h; rw [h, hfc] at hfr; exact lt_irrefl _ hfr
    exact lt_of_le_of_ne hcI.1 hne
  have hcmem : c ∈ Icc a b := ⟨by linarith [hr.1, hcI.1], le_trans hcI.2 hy⟩
  obtain ⟨t, ht, hft⟩ := exists_positive_left_of_negative_derivative
    hrc hfc (hderiv c hcmem hfc) (hneg c hcmem hfc)
  have htcont : ContinuousOn f (Icc r t) :=
    hrcont.mono (fun v hv => ⟨hv.1, by linarith [hv.2, ht.2, hcI.2]⟩)
  obtain ⟨z, hz, hfz⟩ := intermediate_value_Icc ht.1.le htcont ⟨hfr.le, hft.le⟩
  have hzmem : z ∈ zeros := ⟨⟨hz.1, by linarith [hz.2, ht.2, hcI.2]⟩, hfz⟩
  have hcz : c ≤ z := hc.2 hzmem
  linarith [hz.2, ht.2]

/-- A continuous displacement whose zeros all have negative derivative has
at most one zero, including possible interval endpoints. -/
theorem atMostOneZero_of_negative_derivative_at_zeros
    (hcont : ContinuousOn f (Icc a b))
    (hderiv : ∀ x ∈ Icc a b, f x = 0 → HasDerivAt f (f' x) x)
    (hneg : ∀ x ∈ Icc a b, f x = 0 → f' x < 0) :
    AtMostOneZero f (Icc a b) := by
  intro x hx y hy hfx hfy
  rcases lt_trichotomy x y with hxy | heq | hyx
  · exact (no_two_ordered_zeros_of_negative_derivative_at_zeros
      hcont hderiv hneg hx.1 hxy hy.2 hfx hfy).elim
  · exact heq
  · exact (no_two_ordered_zeros_of_negative_derivative_at_zeros
      hcont hderiv hneg hy.1 hyx hx.2 hfy hfx).elim

end OmnibiasAnalytic.Dynamics.Hilbert16Rolle
