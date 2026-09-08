/-
Finite rational stencil obligations (theory 01-11).

The Python bridge emits a list of cross-multiplied equalities `p/q = r/s`
(and, for poisedness, integer Polya comparisons plus a nonzero determinant
witness). This module proves that the Bool evaluators used in
`Omnibias.Generated` mean those finite arithmetic facts.

Scope. The kernel certifies only the algebra of the integers the certificate
carries. That those integers really are the stencil moments / Polya counts /
Vandermonde determinant is a trusted Python input. No collapse, no remainder
bound over a function class, and no continuum statement is expressed here.
-/

import Omnibias.Certificate

namespace Omnibias

/-- Cross-multiplication equality of rationals `p/q` and `r/s`. -/
def ratEq (p q r s : Int) : Bool :=
  decide (p * s = r * q) && decide (q ≠ 0) && decide (s ≠ 0)

theorem ratEq_sound {p q r s : Int} (h : ratEq p q r s = true) :
    p * s = r * q ∧ q ≠ 0 ∧ s ≠ 0 := by
  simp only [ratEq, Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

/-- Conjunction of scaled rational identities. -/
def allRatEq : List (Int × Int × Int × Int) → Bool
  | [] => true
  | t :: rest =>
      ratEq t.1 t.2.1 t.2.2.1 t.2.2.2 && allRatEq rest

theorem allRatEq_sound
    (xs : List (Int × Int × Int × Int))
    (h : allRatEq xs = true)
    {p q r s : Int}
    (hmem : (p, q, r, s) ∈ xs) :
    p * s = r * q ∧ q ≠ 0 ∧ s ≠ 0 := by
  induction xs with
  | nil => cases hmem
  | cons t rest ih =>
      simp only [allRatEq, Bool.and_eq_true] at h
      cases List.mem_cons.mp hmem with
      | inl heq =>
          subst heq
          exact ratEq_sound h.1
      | inr hrest =>
          exact ih h.2 hrest

/-- Polya comparison `needVal ≤ haveVal` on integers. -/
def intGe (haveVal needVal : Int) : Bool := decide (needVal ≤ haveVal)

def allIntGe : List (Int × Int) → Bool
  | [] => true
  | t :: rest => intGe t.1 t.2 && allIntGe rest

theorem allIntGe_sound
    (xs : List (Int × Int))
    (h : allIntGe xs = true)
    {haveVal needVal : Int}
    (hmem : (haveVal, needVal) ∈ xs) : needVal ≤ haveVal := by
  induction xs with
  | nil => cases hmem
  | cons t rest ih =>
      simp only [allIntGe, intGe, Bool.and_eq_true, decide_eq_true_eq] at h
      cases List.mem_cons.mp hmem with
      | inl heq =>
          subst heq
          exact h.1
      | inr hrest =>
          exact ih h.2 hrest

/-- Nonzero rational witness `n/d`. -/
def ratNez (n d : Int) : Bool := decide (n ≠ 0) && decide (d ≠ 0)

theorem ratNez_sound {n d : Int} (h : ratNez n d = true) : n ≠ 0 ∧ d ≠ 0 := by
  simp only [ratNez, Bool.and_eq_true, decide_eq_true_eq] at h
  exact h

/-- Strict inequality of rationals `p/q < r/s` with positive denominators.
Cross-multiplication `p * s < r * q` is the finite residual of a
convergence-ledger margin. Algebra only; no PDE and no analytic class. -/
def ratLt (p q r s : Int) : Bool :=
  decide (0 < q) && decide (0 < s) && decide (p * s < r * q)

theorem ratLt_sound {p q r s : Int} (h : ratLt p q r s = true) :
    0 < q ∧ 0 < s ∧ p * s < r * q := by
  simp only [ratLt, Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

/-- Conjunction of scaled strict rational inequalities. -/
def allRatLt : List (Int × Int × Int × Int) → Bool
  | [] => true
  | t :: rest =>
      ratLt t.1 t.2.1 t.2.2.1 t.2.2.2 && allRatLt rest

theorem allRatLt_sound
    (xs : List (Int × Int × Int × Int))
    (h : allRatLt xs = true)
    {p q r s : Int}
    (hmem : (p, q, r, s) ∈ xs) :
    0 < q ∧ 0 < s ∧ p * s < r * q := by
  induction xs with
  | nil => cases hmem
  | cons t rest ih =>
      simp only [allRatLt, Bool.and_eq_true] at h
      cases List.mem_cons.mp hmem with
      | inl heq =>
          subst heq
          exact ratLt_sound h.1
      | inr hrest =>
          exact ih h.2 hrest

end Omnibias
