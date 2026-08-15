/-
Rational interval arithmetic and a finite enclosure-trace interpreter.

`QInterval` is the `ℚ` analogue of the Mathlib-free kernel's `ZInterval`.
`evalTrace` replays a finite DAG of `+ − × abs recip` on point or box
intervals. A green build certifies the arithmetic of that DAG.

Scope (honest). This is finite rational interval arithmetic on a named
trace. It is not a continuum PDE claim, and it does not evaluate
transcendental functions.
-/

import Mathlib.Tactic

namespace OmnibiasAnalytic.Check

/-- A closed rational interval `[lo, hi]`. Well-formedness (`lo ≤ hi`) is a
separate proposition, matching the kernel `ZInterval` shape. -/
structure QInterval where
  lo : ℚ
  hi : ℚ
deriving Repr, DecidableEq, Inhabited

namespace QInterval

/-- Membership: `lo ≤ x ≤ hi`. -/
def Mem (x : ℚ) (I : QInterval) : Prop := I.lo ≤ x ∧ x ≤ I.hi

/-- A nonempty (well-formed) interval has `lo ≤ hi`. -/
def Valid (I : QInterval) : Prop := I.lo ≤ I.hi

/-- A collapsed (point) interval. -/
def point (q : ℚ) : QInterval := ⟨q, q⟩

@[simp] theorem point_lo (q : ℚ) : (point q).lo = q := rfl
@[simp] theorem point_hi (q : ℚ) : (point q).hi = q := rfl

theorem mem_point (q : ℚ) : Mem q (point q) := ⟨le_rfl, le_rfl⟩

/-- Interval addition `[a.lo+b.lo, a.hi+b.hi]`. -/
def add (a b : QInterval) : QInterval := ⟨a.lo + b.lo, a.hi + b.hi⟩

/-- Interval negation `[-a.hi, -a.lo]`. -/
def neg (a : QInterval) : QInterval := ⟨-a.hi, -a.lo⟩

/-- Interval subtraction `a - b := a + (-b)`. -/
def sub (a b : QInterval) : QInterval := add a (neg b)

/-- Interval multiplication: the tightest interval containing the four
corner products. -/
def mul (a b : QInterval) : QInterval :=
  ⟨min (min (a.lo * b.lo) (a.lo * b.hi)) (min (a.hi * b.lo) (a.hi * b.hi)),
   max (max (a.lo * b.lo) (a.lo * b.hi)) (max (a.hi * b.lo) (a.hi * b.hi))⟩

/-- Interval absolute value. -/
def absI (a : QInterval) : QInterval :=
  if 0 ≤ a.lo then a
  else if a.hi ≤ 0 then ⟨-a.hi, -a.lo⟩
  else ⟨0, max (-a.lo) a.hi⟩

/-- Reciprocal when `0` is outside the interval; `none` otherwise. -/
def recip? (a : QInterval) : Option QInterval :=
  if 0 < a.lo ∨ a.hi < 0 then some ⟨a.hi⁻¹, a.lo⁻¹⟩ else none

theorem mem_add {x y : ℚ} {a b : QInterval} (hx : Mem x a) (hy : Mem y b) :
    Mem (x + y) (add a b) := by
  simp only [Mem, add] at hx hy ⊢
  constructor <;> linarith

theorem mem_neg {x : ℚ} {a : QInterval} (hx : Mem x a) : Mem (-x) (neg a) := by
  simp only [Mem, neg] at hx ⊢
  constructor <;> linarith

theorem mem_sub {x y : ℚ} {a b : QInterval} (hx : Mem x a) (hy : Mem y b) :
    Mem (x - y) (sub a b) := by
  unfold sub
  simpa [sub_eq_add_neg] using mem_add hx (mem_neg hy)

theorem mul_mem_1dR {p q t c : ℚ} (h1 : p ≤ t) (h2 : t ≤ q) :
    min (p * c) (q * c) ≤ t * c ∧ t * c ≤ max (p * c) (q * c) := by
  rcases le_total (0 : ℚ) c with hc | hc
  · have hpt := mul_le_mul_of_nonneg_right h1 hc
    have htq := mul_le_mul_of_nonneg_right h2 hc
    have hpq := mul_le_mul_of_nonneg_right (h1.trans h2) hc
    rw [min_eq_left hpq, max_eq_right hpq]
    exact ⟨hpt, htq⟩
  · have htp := mul_le_mul_of_nonpos_right h1 hc
    have hqt := mul_le_mul_of_nonpos_right h2 hc
    have hqp := mul_le_mul_of_nonpos_right (h1.trans h2) hc
    rw [min_eq_right hqp, max_eq_left hqp]
    exact ⟨hqt, htp⟩

theorem mul_mem_1dL {p q t c : ℚ} (h1 : p ≤ t) (h2 : t ≤ q) :
    min (c * p) (c * q) ≤ c * t ∧ c * t ≤ max (c * p) (c * q) := by
  simpa [mul_comm] using mul_mem_1dR (p := p) (q := q) (t := t) (c := c) h1 h2

theorem mem_mul {x y : ℚ} {a b : QInterval} (hx : Mem x a) (hy : Mem y b) :
    Mem (x * y) (mul a b) := by
  have s1 := mul_mem_1dR (p := a.lo) (q := a.hi) (t := x) (c := y) hx.1 hx.2
  have s2 := mul_mem_1dL (p := b.lo) (q := b.hi) (t := y) (c := a.lo) hy.1 hy.2
  have s3 := mul_mem_1dL (p := b.lo) (q := b.hi) (t := y) (c := a.hi) hy.1 hy.2
  refine ⟨?lo, ?hi⟩
  · have hxy : min (a.lo * y) (a.hi * y) ≤ x * y := s1.1
    have hcorners :
        min (min (a.lo * b.lo) (a.lo * b.hi)) (min (a.hi * b.lo) (a.hi * b.hi)) ≤
          min (a.lo * y) (a.hi * y) := min_le_min s2.1 s3.1
    exact hcorners.trans hxy
  · have hxy : x * y ≤ max (a.lo * y) (a.hi * y) := s1.2
    have hcorners :
        max (a.lo * y) (a.hi * y) ≤
          max (max (a.lo * b.lo) (a.lo * b.hi)) (max (a.hi * b.lo) (a.hi * b.hi)) :=
      max_le_max s2.2 s3.2
    exact hxy.trans hcorners

theorem mem_abs {x : ℚ} {a : QInterval} (hx : Mem x a) : Mem |x| (absI a) := by
  unfold absI
  split_ifs with hlo hhi
  · have hx0 : 0 ≤ x := hlo.trans hx.1
    rwa [abs_of_nonneg hx0]
  · have hx0 : x ≤ 0 := hx.2.trans hhi
    rw [abs_of_nonpos hx0]
    exact ⟨neg_le_neg hx.2, neg_le_neg hx.1⟩
  · refine ⟨abs_nonneg x, ?_⟩
    have hxlo : -x ≤ -a.lo := neg_le_neg hx.1
    have h1 : x ≤ max (-a.lo) a.hi := hx.2.trans (le_max_right _ _)
    have h2 : -x ≤ max (-a.lo) a.hi := hxlo.trans (le_max_left _ _)
    exact (abs_le.mpr ⟨neg_le.mp h2, h1⟩)

theorem mem_recip {x : ℚ} {a : QInterval} (hx : Mem x a)
    (h : 0 < a.lo ∨ a.hi < 0) : Mem x⁻¹ ⟨a.hi⁻¹, a.lo⁻¹⟩ := by
  rcases h with hpos | hneg
  · have hxpos : 0 < x := lt_of_lt_of_le hpos hx.1
    have hahi : 0 < a.hi := lt_of_lt_of_le hpos (hx.1.trans hx.2)
    constructor
    · exact (inv_le_inv₀ hahi hxpos).mpr hx.2
    · exact (inv_le_inv₀ hxpos hpos).mpr hx.1
  · have hxneg : x < 0 := lt_of_le_of_lt hx.2 hneg
    have halo : a.lo < 0 := lt_of_le_of_lt hx.1 hxneg
    constructor
    · exact (inv_le_inv_of_neg hneg hxneg).mpr hx.2
    · exact (inv_le_inv_of_neg hxneg halo).mpr hx.1

theorem pos_of_mem_of_lo_pos {x : ℚ} {a : QInterval} (hx : Mem x a) (h : 0 < a.lo) :
    0 < x :=
  lt_of_lt_of_le h hx.1

theorem eq_of_mem_point {x : ℚ} {a : QInterval} (hx : Mem x a) (h : a.hi ≤ a.lo) :
    x = a.lo :=
  le_antisymm (hx.2.trans h) hx.1

end QInterval

/-! ### Finite enclosure traces -/

/-- One node of a replayable rational enclosure DAG. Indices refer to
previously emitted nodes (`0`-based). -/
inductive TraceOp where
  | const (q : ℚ)
  | add (i j : ℕ)
  | sub (i j : ℕ)
  | mul (i j : ℕ)
  | abs (i : ℕ)
  | recip (i : ℕ)
deriving Repr, DecidableEq, Inhabited

open QInterval

/-- Evaluate one op against the nodes already produced. -/
def evalOne (acc : List QInterval) : TraceOp → Option QInterval
  | .const q => some (point q)
  | .add i j =>
    match acc[i]?, acc[j]? with
    | some a, some b => some (a.add b)
    | _, _ => none
  | .sub i j =>
    match acc[i]?, acc[j]? with
    | some a, some b => some (a.sub b)
    | _, _ => none
  | .mul i j =>
    match acc[i]?, acc[j]? with
    | some a, some b => some (a.mul b)
    | _, _ => none
  | .abs i =>
    match acc[i]? with
    | some a => some a.absI
    | none => none
  | .recip i =>
    match acc[i]? with
    | some a => a.recip?
    | none => none

/-- Replay `ops` from left to right. A bad index or a reciprocal through
zero aborts and returns the empty list. -/
def evalTraceGo (acc : List QInterval) : List TraceOp → List QInterval
  | [] => acc
  | op :: rest =>
    match evalOne acc op with
    | some I => evalTraceGo (acc ++ [I]) rest
    | none => []

def evalTrace (ops : List TraceOp) : List QInterval :=
  evalTraceGo [] ops

end OmnibiasAnalytic.Check
