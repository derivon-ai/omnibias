# LN/exp cell test for the coalescing quadratic passage

This note tests the bounded-format Log-Noetherian/exp proposal against the
actual selected quadratic passage of
[the coalescing atlas](HILBERT16-COALESCING-CAPTURE.md).  It keeps the
existing physical sections and the small-label connector
`|t_i| <= tbox < u^2/2`, `0 < u < 2`.  It does not introduce a new closing
map.

The test has a negative outcome.  The two elementary logarithmic-derivative
lemmas are valid, but they do not prove that the physical first-hit map belongs
to a bounded-format LN core.  Two exact checks locate the failure:

1. the proposed super-small-separation tuple with `L = 1`, `lambda1 = -3`
   is not in the quadratic family; after correcting it, the outgoing
   `W`-ratio still diverges;
2. the proposed shrinking-root radius `a = r1 - theta*sep` becomes negative
   for every fixed `theta > 0`.

Consequently the factorization below remains a candidate analytic obligation,
not a proved LN membership theorem.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. Notation lock

Several earlier notes use the same letter in different charts.  This note uses
the following meanings.

| Symbol | Meaning here | Other use not imported here |
| --- | --- | --- |
| `u_conn` | fixed incoming connector coordinate, `0 < u_conn < 2` | `u_sc = epsilon/omega` in the fixed-product chart |
| `x` | outgoing coordinate `V = -epsilon*x` | a generic real variable |
| `kappa_h` | inverse logarithmic centre height, `H = epsilon^3 exp(-kappa_h/epsilon)` | `kappa_field = -g` in the vector field |
| `sigma_bl` | a blow-up scale | a sign or activation |
| `sep` | nonnegative root separation | never an independent coefficient |
| `Delta` | `4L - lambda1^2` | opposite sign to `delta = lambda1^2 - 4L` |

Thus

    delta = lambda1^2 - 4L = sep^2,    Delta = -delta = -sep^2,
    r1 = (-lambda1 - sep)/2,           2*r1 + sep + lambda1 = 0.

The relation `sep^2 = lambda1^2 - 4L` is part of the parameter space.  A
purported cell point that violates it is rejected rather than treated as an
independent unfolding.

The exact normal field and outgoing chart are

    Vdot = f(V) + h*g(V,h),                 hdot = -V*h,
    f(V) = -L*epsilon^3 + lambda1*epsilon^2*V
           + epsilon*V^2*zeta(V,epsilon),
    V = -epsilon*x,
    B_epsilon(x) = L + lambda1*x - x^2*zeta(-epsilon*x,epsilon).

At `epsilon = 0`, `zeta = -1` at the centre and
`B_-(x) = L + lambda1*x + x^2`.

## 2. What the proposed factorization would have to mean

Let `D_phys` be the actual first-hit map between the already selected
physical sections, on the subset on which both hits exist and are transverse.
Let `E_in` and `E_out` be specified invertible exp/log coordinate changes.
On that subset one may define

    Phi = E_out^{-1} composed with D_phys composed with E_in^{-1},

and hence obtain the exact tautological identity

    D_phys = E_out composed with Phi composed with E_in.        (F)

Identity (F) is not the missing theorem.  G1 needs all of the following:

- one parameter-independent finite cell cover of the declared physical
  domain;
- existence and transversality of `D_phys` on every cell;
- holomorphic extension of `Phi` to a uniform relative enlargement;
- exact differential-polynomial closure of the functions that include
  `Phi`, not merely of coordinate functions chosen in advance;
- a format and sup-norm bound independent of
  `(epsilon, sep, L, lambda1)`;
- equality of the independently constructed physical maps on overlaps.

The existing estimates provide upper bounds on variational factors.  An upper
bound is not an exact factor of `D_phys`, so it cannot be peeled off as
`E_out` in (F).  Conversely, defining `Phi` by (F) proves no LN membership.
This separates the useful coordinate idea from the unsupported analytic step.

## 3. L1: logarithmic-chart Cauchy bound

**Lemma L1.**  Let `0 < delta_ext < 1`, `0 < r1 < r2`, and let `f` be
holomorphic on

    A(delta_ext*r1, r2/delta_ext)

with `|f| <= M`.  Then for every integer `k >= 0`, on `A(r1,r2)`,

    |(z d/dz)^k f(z)|
        <= k! M / log(1/delta_ext)^k.                       (L1)

For `k = 2` this is

    |(z d/dz)^2 f(z)|
        <= 2M / log(1/delta_ext)^2.

**Proof.**  Lift the annulus to its logarithmic universal cover.  Put
`F(w) = f(exp(w))`.  It is holomorphic and `2*pi*i`-periodic on the strip

    log(delta_ext*r1) < Re(w) < log(r2/delta_ext),

and `|F| <= M`.  If `r1 < |z| < r2` and `w` is any logarithm of `z`, the
Euclidean distance from `w` to each vertical boundary of the enlarged strip
is at least

    rho = log(1/delta_ext).

For every `0 < s < rho`, the closed circle `|xi-w|=s` lies in the enlarged
strip.  Cauchy's derivative formula gives

    |F^(k)(w)| <= k! M / s^k.

Letting `s` increase to `rho` proves the displayed bound.  Finally the chain
rule gives `d/dw f(exp(w)) = (z d/dz)f(z)` and, by induction,
`F^(k)(w) = (z d/dz)^k f(z)`.  This proves (L1).  The proof is independent of
any LN preprint and includes the factorial.

L1 is conditional on the actual map function having the stated holomorphic
extension and sup bound.  A cell declaration does not establish either
hypothesis.

## 4. L2: monomial eigenvalue

**Lemma L2.**  On a chosen branch on which `z^alpha` is holomorphic,

    (z d/dz)^k (c*z^alpha) = alpha^k*c*z^alpha             (L2)

for every integer `k >= 0`.

**Proof.**  For `k=0` the statement is the identity.  If it holds at `k`,
then

    z d/dz [alpha^k*c*z^alpha]
      = alpha^k*c*alpha*z^alpha
      = alpha^(k+1)*c*z^alpha.

Induction proves the claim.  Integer exponents are single-valued on a
punctured disc or annulus.  For noninteger real or complex `alpha`, a branch
of `log z` and the corresponding simply connected chart must be part of the
data.

L2 controls a logarithmic derivative relative to the size of the monomial.
It does not bound `|c*z^alpha|` itself.  An unbounded monomial therefore still
has an unbounded absolute second derivative even though its exponent is
fixed.

## 5. Candidate cells and the first exact failure

The proposal names three real cells:

- `cellA`, adapted to `sep -> 0`, with
  `tau = epsilon*log(1/sep)` or `W = h^epsilon/epsilon`;
- `cellB`, adapted to `r1 -> 0`, with a radial radius depending on the base;
- `cellAB`, on which both descriptions have positive radii and are compared
  using the same physical labels.

These names form a finite ledger, but not yet a cover by bounded-format LN
maps.  On the intended super-small sequence,

    epsilon_n = 1/n,    sep_n = exp(-1/epsilon_n^2),
    lambda1 = -3,

the quadratic relation forces

    L_n = (9 - sep_n^2)/4 -> 9/4.                          (K-A)

The tuple printed in the proposal with `L=1` instead has
`sep=sqrt(5)` and is not a coalescing sequence.  The corrected path (K-A)
is inside the first-root family and retains the intended obstruction.

Indeed, for a separation-scale exit
`h_e = c*epsilon^3*sep^2` and any existing outgoing section
`h_max = c_max*epsilon^N`, `N` fixed,

    log(W_max/W_e)
      = epsilon*log(h_max/h_e)
      = 2/epsilon + (N-3)*epsilon*log(epsilon) + O(epsilon).

Thus `W_max/W_e -> infinity`.  Equivalently,
`tau = epsilon*log(1/sep) = 1/epsilon -> infinity`.
Making either quantity a coordinate does not give it a parameter-independent
sup norm.  L2 only says that a fixed-power monomial is an eigenfunction of the
logarithmic derivation; its absolute bound still contains this divergent
value.

The precise failed `cellA` obligation is therefore:

> no uniform sup bound has been proved for the physical matching function
> carrying the separation-scale exit to any existing outgoing section, and
> the available variational expression is a bound rather than an exact exp
> chart factor.

## 6. Shrinking-root radius and the second exact failure

On the second named path,

    L_n = 1/n,    lambda1 = -2,
    sep_n = sqrt(4-4/n),
    r1_n = (2-sep_n)/2 -> 0.

The proposed `cellB` radius is

    a_n = r1_n - theta*sep_n.

For every fixed `theta > 0`, `a_n -> -2*theta`; hence it is eventually
negative.  It is not an LN fiber radius on this path and has no positive
`delta_ext` margin.  This is a direct sign failure, before any Cauchy or
format estimate.

A legitimate relative wall could be written

    a_rel = (1-theta)*r1,    0 < theta < 1.

It stays positive for each finite `n`, but tends to zero in physical
coordinates.  Relative positivity does not restore the absolute
transversality estimate `|V| >= epsilon*a_min` used by the existing outgoing
continuation.  A proof that the rescaled orbit reaches the fixed large section
with uniform first and second parameter variations is still required.

The precise failed `cellB` obligation is therefore the proposed radius
function itself; after replacing it by a positive relative radius, physical
outgoing first-hit and its uniform variational bounds remain open.

## 7. Admission and overlap

A finite compatibility table can check that a declared guard polynomial is
identically zero or has a certified nonzero interval range on a declared
box.  It can also refuse an ambiguous guard.  Such a check is useful
bookkeeping, but it is not first-hit completeness: the latter additionally
requires existence up to the section, absence of an earlier hit, and a
uniformly nonzero event derivative for the actual ODE.

The regular finite arc may be certified by `certify_stopped_event` when its
source, tube, time, and event margins are supplied.  That certificate cannot
be composed across the singular tail until the tail map exists on a
compatible physical domain.  Similarly, equality of chart coordinate
formulas on `cellAB` does not prove equality of two unconstructed physical
continuations.

The incoming small-label tube remains exactly the one already justified:

    |t_i| <= tbox < u_conn^2/2,    0 < u_conn < 2.

No stronger admission claim is inferred from cell membership.

## 8. G1 obligation map

| G1 item | Written analytic part | Finite-certificate part | Verdict |
| --- | --- | --- | --- |
| 1. Finite weighted chart cover | Exact quadratic relation and candidate `cellA/cellB/cellAB` coordinates | rational identities, radius signs, format budgets | **open**: corrected kill A has unbounded matching data; proposed `cellB` radius is negative |
| 2. Complete physical first-hit | existing incoming connector and regular stopped-event theorem | guard compatibility/refusal table | **open**: no singular outgoing first-hit theorem on shrinking-root path |
| 3. Uniform derivatives | L1 and L2 above | exact-Q chain replay and interval sup bounds for declared surrogate functions | **open**: actual `Phi` membership, extension, sup bound, and `C^2` remainder are absent |
| 4. Endpoint matching | physical uniqueness would identify maps once both continuations exist | coordinate overlap identities | **open**: the required continuations are not both constructed on the limiting overlaps |

The finite certificates implemented with this note replay only the entries in
the third column.  They cannot turn an open entry in the second column into a
theorem.

## 9. Collapse and formal scope

No new named collapse is introduced.  A sound enclosure whose width tends to
zero is already the repository's enclosure collapse.  Exact-Q chain closure,
format inequalities, and radius signs are finite obligations; Lean may replay
them.  Lean does not prove that the physical ODE solution equals the declared
chain unless that analytic statement is separately formalized.

The valid result of this increment is therefore a sharper obstruction:

> Logarithmic coordinates remove neither the absolute sup-norm requirement
> on corrected kill A nor the physical transversality loss on kill B.  The
> proposed fixed `theta` wall fails even as a positive radius.

G1 remains false.  This is not a counterexample to finite cyclicity and not a
claim about Hilbert's sixteenth problem beyond the declared passage.

The executable GL1–GL7 replay confirms this assessment through both named
sequences up to `n=2000`.  The exact coordinate chain, supplied finite sup
enclosures, negative controls, and regular stopped-event model pass; the
`ln_chain_closure` and `ln_format_bound` rational obligations receive genuine
minimal-kernel Lean verification.  The artifact still records
`physical_phi_member=false`, `physical_uniform_c2=false`,
`physical_first_hit_complete=false`, and `g1_passed=false`.

The follow-up
[direct format barrier](HILBERT16-LN-FORMAT-BARRIER.md) makes the missing
uniformity quantitative: finite `tau`/log-W chains close exactly, while their
outer radius and chain sup norm grow linearly on corrected kill A.  A
zero-preserving normalized physical-return chain remains open.
