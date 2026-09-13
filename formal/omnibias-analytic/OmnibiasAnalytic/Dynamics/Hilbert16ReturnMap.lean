/-
Return-event algebra and actual-derivative zero-count implications.

The event chain-rule equations and derivative hypotheses are assumptions.
This module does not assert existence of a physical return, soundness of a
numerical flow tube, or capture of all cycles of a graphic.
-/
import OmnibiasAnalytic.Dynamics.Hilbert16Rolle
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

namespace OmnibiasAnalytic.Dynamics.Hilbert16ReturnMap

open Set Hilbert16Rolle

/-- Dividing by a real exponential preserves exactly the zero set. -/
theorem exponential_weight_zero_iff (a x value : ℝ) :
    Real.exp (-a * x) * value = 0 ↔ value = 0 := by
  exact mul_eq_zero_iff_left (Real.exp_ne_zero _)

/-- The derivative-division operation used by the finite exponential class.
It requires the actual derivative; no finite Taylor prefix is substituted. -/
theorem exponential_weight_derivative
    {f : ℝ → ℝ} {a x value : ℝ} (hf : HasDerivAt f value x) :
    HasDerivAt (fun t => Real.exp (-a * t) * f t)
      (Real.exp (-a * x) * (value - a * f x)) x := by
  have he : HasDerivAt (fun t => Real.exp (-a * t))
      (Real.exp (-a * x) * (-a * 1)) x :=
    (Real.hasDerivAt_exp (-a * x)).comp x ((hasDerivAt_id x).const_mul (-a))
  apply (he.mul hf).congr_deriv
  ring

/-- First event-time sensitivity from its actual chain-rule equation. -/
theorem event_time_first
    {qt qa ta : ℝ} (hqt : qt ≠ 0) (hchain : qa + qt * ta = 0) :
    ta = -qa / qt := by
  apply (eq_div_iff hqt).2
  linarith

/-- Mixed event-time sensitivity; both event-time acceleration terms remain. -/
theorem event_time_mixed
    {qt qab qat qbt qtt ta tb tab : ℝ} (hqt : qt ≠ 0)
    (hchain : qab + qat * tb + qbt * ta + qtt * ta * tb + qt * tab = 0) :
    tab = -(qab + qat * tb + qbt * ta + qtt * ta * tb) / qt := by
  apply (eq_div_iff hqt).2
  linarith

/-- A nonzero actual first derivative excludes two displacement zeros. -/
theorem atMostOneZero_of_nonzero_derivative
    {f f' : ℝ → ℝ} {a b : ℝ}
    (hc : ContinuousOn f (Icc a b))
    (hd : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x)
    (hn : ∀ x ∈ Ioo a b, f' x ≠ 0) :
    AtMostOneZero f (Icc a b) := by
  intro x hx y hy hfx hfy
  by_contra hne
  rcases lt_or_gt_of_ne hne with hxy | hyx
  · obtain ⟨c, hcxy, hzero⟩ := exists_derivative_zero_between hc hd
      hx.1 hxy hy.2 hfx hfy
    exact hn c ⟨lt_of_le_of_lt hx.1 hcxy.1, lt_of_lt_of_le hcxy.2 hy.2⟩ hzero
  · obtain ⟨c, hcyx, hzero⟩ := exists_derivative_zero_between hc hd
      hy.1 hyx hx.2 hfy hfx
    exact hn c ⟨lt_of_le_of_lt hy.1 hcyx.1, lt_of_lt_of_le hcyx.2 hx.2⟩ hzero

/-- A nonzero actual second derivative excludes three ordered zeros. -/
theorem no_three_zeros_of_nonzero_second_derivative
    {f f' f'' : ℝ → ℝ} {a b : ℝ}
    (hc : ContinuousOn f (Icc a b))
    (hc' : ContinuousOn f' (Icc a b))
    (hd : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x)
    (hd' : ∀ x ∈ Ioo a b, HasDerivAt f' (f'' x) x)
    (hn : ∀ x ∈ Ioo a b, f'' x ≠ 0)
    {x y z : ℝ} (ha : a ≤ x) (hxy : x < y) (hyz : y < z) (hb : z ≤ b)
    (hfx : f x = 0) (hfy : f y = 0) (hfz : f z = 0) : False := by
  have hone := atMostOneZero_of_nonzero_derivative hc' hd' hn
  apply no_three_ordered_zeros hc hd ?_ ha hxy hyz hb hfx hfy hfz
  intro u hu v hv hfu hfv
  exact hone u ⟨hu.1.le, hu.2.le⟩ v ⟨hv.1.le, hv.2.le⟩ hfu hfv

/-- A strictly signed interval enclosure excludes zero. -/
theorem nonzero_of_signed_enclosure
    {lo hi value : ℝ} (hl : lo ≤ value) (hh : value ≤ hi)
    (hs : 0 < lo ∨ hi < 0) : value ≠ 0 := by
  rcases hs with hs | hs <;> intro hz <;> linarith

/-- The identically zero function has no isolated zero on an open interval. -/
theorem identity_has_nearby_zero
    {f : ℝ → ℝ} {a b x radius : ℝ}
    (hx : x ∈ Ioo a b) (hr : 0 < radius)
    (hid : ∀ y ∈ Ioo a b, f y = 0) :
    ∃ y ∈ Ioo a b, y ≠ x ∧ |y - x| < radius ∧ f y = 0 := by
  let d := min ((b - x) / 2) (radius / 2)
  have hd : 0 < d := lt_min (by linarith [hx.2]) (by linarith)
  have hdb : d ≤ (b - x) / 2 := min_le_left _ _
  have hdr : d ≤ radius / 2 := min_le_right _ _
  have hy : x + d ∈ Ioo a b := ⟨by linarith [hx.1], by linarith [hx.2]⟩
  refine ⟨x + d, hy, ?_, ?_, hid _ hy⟩
  · linarith
  · rw [show x + d - x = d by ring, abs_of_pos hd]
    linarith

end OmnibiasAnalytic.Dynamics.Hilbert16ReturnMap
