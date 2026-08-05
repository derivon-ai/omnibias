/-
Sound integer-interval arithmetic (Mathlib-free).

`ZInterval` is a closed interval `[lo, hi]` of `Int`.  Rational endpoints are
represented by scaling to a common positive denominator *before* entering the
kernel, so the kernel itself only manipulates exact `Int` values and every
soundness lemma is discharged by the core `omega` decision procedure -- no
`sorry`, no Mathlib.

Soundness lemma shape: if `x ∈ a` and `y ∈ b` then `x ⊕ y ∈ (a ⊕ b)` for each
kernel operation `⊕`.  These are the lemmas a certificate checker chains to turn
the (finite, rational) enclosure data of an omnibias certificate into a
kernel-checked sign / gap theorem.
-/

namespace Omnibias

/-- A closed integer interval `[lo, hi]`. -/
structure ZInterval where
  lo : Int
  hi : Int
deriving Repr, DecidableEq

namespace ZInterval

/-- Membership: `lo ≤ x ≤ hi`. -/
def Mem (x : Int) (I : ZInterval) : Prop := I.lo ≤ x ∧ x ≤ I.hi

/-- A nonempty (well-formed) interval has `lo ≤ hi`. -/
def Valid (I : ZInterval) : Prop := I.lo ≤ I.hi

/-- Interval addition `[a.lo+b.lo, a.hi+b.hi]`. -/
def add (a b : ZInterval) : ZInterval := ⟨a.lo + b.lo, a.hi + b.hi⟩

/-- Interval negation `[-a.hi, -a.lo]`. -/
def neg (a : ZInterval) : ZInterval := ⟨-a.hi, -a.lo⟩

/-- Interval subtraction `a - b := a + (-b)`. -/
def sub (a b : ZInterval) : ZInterval := add a (neg b)

/-- Interval multiplication.  The product of `[a.lo, a.hi]` and `[b.lo, b.hi]` is the
tightest interval containing all four corner products; its endpoints are their min and
max.  Uses only `+`/`*`/`min`/`max` on `Int` (no division), so it stays inside the
Mathlib-free kernel. -/
def mul (a b : ZInterval) : ZInterval :=
  ⟨min (min (a.lo * b.lo) (a.lo * b.hi)) (min (a.hi * b.lo) (a.hi * b.hi)),
   max (max (a.lo * b.lo) (a.lo * b.hi)) (max (a.hi * b.lo) (a.hi * b.hi))⟩

/-- Soundness of addition. -/
theorem mem_add {x y : Int} {a b : ZInterval} (hx : Mem x a) (hy : Mem y b) :
    Mem (x + y) (add a b) := by
  simp only [Mem, add] at hx hy ⊢
  omega

/-- Soundness of negation. -/
theorem mem_neg {x : Int} {a : ZInterval} (hx : Mem x a) : Mem (-x) (neg a) := by
  simp only [Mem, neg] at hx ⊢
  omega

/-- Soundness of subtraction. -/
theorem mem_sub {x y : Int} {a b : ZInterval} (hx : Mem x a) (hy : Mem y b) :
    Mem (x - y) (sub a b) := by
  simp only [Mem, sub, add, neg] at hx hy ⊢
  omega

/-- One-dimensional monotone bound, fixed factor on the **right**: for `p ≤ t ≤ q`, the
product `t * c` lies between `p * c` and `q * c` (which is the lower/upper endpoint is
decided by the sign of `c`, handled by the two monotonicity lemmas). -/
theorem mul_mem_1dR {p q t c : Int} (h1 : p ≤ t) (h2 : t ≤ q) :
    min (p * c) (q * c) ≤ t * c ∧ t * c ≤ max (p * c) (q * c) := by
  cases Int.le_total 0 c with
  | inl hc =>
      have hpt : p * c ≤ t * c := Int.mul_le_mul_of_nonneg_right h1 hc
      have htq : t * c ≤ q * c := Int.mul_le_mul_of_nonneg_right h2 hc
      omega
  | inr hc =>
      have htp : t * c ≤ p * c := Int.mul_le_mul_of_nonpos_right h1 hc
      have hqt : q * c ≤ t * c := Int.mul_le_mul_of_nonpos_right h2 hc
      omega

/-- One-dimensional monotone bound, fixed factor on the **left**: for `p ≤ t ≤ q`, the
product `c * t` lies between `c * p` and `c * q`. -/
theorem mul_mem_1dL {p q t c : Int} (h1 : p ≤ t) (h2 : t ≤ q) :
    min (c * p) (c * q) ≤ c * t ∧ c * t ≤ max (c * p) (c * q) := by
  cases Int.le_total 0 c with
  | inl hc =>
      have hpt : c * p ≤ c * t := Int.mul_le_mul_of_nonneg_left h1 hc
      have htq : c * t ≤ c * q := Int.mul_le_mul_of_nonneg_left h2 hc
      omega
  | inr hc =>
      have htp : c * t ≤ c * p := Int.mul_le_mul_of_nonpos_left hc h1
      have hqt : c * q ≤ c * t := Int.mul_le_mul_of_nonpos_left hc h2
      omega

/-- **Soundness of multiplication.**  If `x ∈ a` and `y ∈ b` then `x * y ∈ a * b`.  This
is the one genuinely nonlinear kernel lemma: `omega` cannot see through `x * y`, so the
proof factors the bound through two one-dimensional monotone steps (`mul_mem_1dR` on the
varying `x` with `y` fixed, then `mul_mem_1dL` on the varying `y` with each `a`-endpoint
fixed), leaving `omega` only the linear `min`/`max` chaining over the corner products. -/
theorem mem_mul {x y : Int} {a b : ZInterval} (hx : Mem x a) (hy : Mem y b) :
    Mem (x * y) (mul a b) := by
  simp only [Mem, mul]
  simp only [Mem] at hx hy
  have s1 := mul_mem_1dR (c := y) hx.1 hx.2
  have s2 := mul_mem_1dL (c := a.lo) hy.1 hy.2
  have s3 := mul_mem_1dL (c := a.hi) hy.1 hy.2
  omega

/-- Validity is preserved by multiplication (the min of the corner products never
exceeds their max, unconditionally). -/
theorem valid_mul {a b : ZInterval} (_ha : Valid a) (_hb : Valid b) : Valid (mul a b) := by
  simp only [Valid, mul]
  omega

/-- Validity is preserved by addition. -/
theorem valid_add {a b : ZInterval} (ha : Valid a) (hb : Valid b) : Valid (add a b) := by
  simp only [Valid, add] at ha hb ⊢
  omega

/-- Validity is preserved by negation. -/
theorem valid_neg {a : ZInterval} (ha : Valid a) : Valid (neg a) := by
  simp only [Valid, neg] at ha ⊢
  omega

/-- **Positivity certificate.** A value drawn from an interval whose lower
endpoint is positive is itself strictly positive. -/
theorem pos_of_mem_of_lo_pos {x : Int} {a : ZInterval} (hx : Mem x a) (h : 0 < a.lo) :
    0 < x := by
  simp only [Mem] at hx
  omega

/-- A value drawn from an interval with nonnegative lower endpoint is nonnegative. -/
theorem nonneg_of_mem_of_lo_nonneg {x : Int} {a : ZInterval} (hx : Mem x a) (h : 0 ≤ a.lo) :
    0 ≤ x := by
  simp only [Mem] at hx
  omega

/-- A value drawn from an interval whose upper endpoint is negative is negative
(the dual obligation used to *exclude* a property, e.g. a CLM non-blow-up). -/
theorem neg_of_mem_of_hi_neg {x : Int} {a : ZInterval} (hx : Mem x a) (h : a.hi < 0) :
    x < 0 := by
  simp only [Mem] at hx
  omega

/-- **Equality certificate.** A value drawn from an interval that has collapsed to a
point (`a.hi ≤ a.lo`, hence `a.lo = a.hi` for a valid interval) equals that point.
This is the `ZInterval` primitive for a rational *equality* obligation: a
special-number identity `p/q = r/s`, scaled to a common positive `Int` denominator,
reduces to the exact `Int` equality `p·s − r·q = 0`, i.e. that difference lies in the
point interval `[0, 0]`. Unlike the sign lemmas this concludes an equality, so a
rational identity -- not just a sign -- is kernel-checkable. -/
theorem eq_of_mem_point {x : Int} {a : ZInterval} (hx : Mem x a) (h : a.hi ≤ a.lo) :
    x = a.lo := by
  simp only [Mem] at hx
  omega

end ZInterval
end Omnibias
