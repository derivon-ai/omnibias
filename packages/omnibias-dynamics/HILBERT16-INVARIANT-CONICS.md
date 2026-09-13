# Local invariant conics in the full quadratic family

Write `p=mu2`, `r=mu3`, `h=mu1`, `alpha=A−1`, and fix any `C0>1`. All statements below concern a sufficiently small neighborhood of `(A,h,p,r,C)=(1,0,0,0,C0)` and the normalized conic

`F=a x²+bxy+d y²+e x+y+f`, near `(a,b,d,e,f)=(-1/2,0,0,0,C/2)`.

The vector field is

`P=A x−y+x²+(p+r)xy+h y²`, `Q=C x+x²+xy+r y²`.

**Result:** the local invariant-conic solutions form the union of exactly two real-analytic graphs over `(p,r,C)`. One consists of parabolas and has the proposed compensation

`h=p(2p+r)+O((|p|+|r|)³)`, `alpha=−(C+6)p−3r+O((|p|+|r|)²)`.

The other is generally nonparabolic and has

`h=−2pr+O((|p|+|r|)³)`, `alpha=Cp−3r+O((|p|+|r|)²)`.

This is a local algebraic classification of invariant conics, not a theorem about all periodic orbits, the full return map, or Hilbert's problem. No novelty relative to the invariant-conic literature is asserted. The coefficient identities and determinants below have exact symbolic replay; the analytic conclusion uses the ordinary implicit function theorem.

## Exact reduction and completeness

A nearby nonsingular conic is irreducible and has a smooth real arc. If the
field is tangent along that arc, the polynomials F and P F_x+Q F_y have
infinitely many common points. Bezout's theorem then forces F to divide the
latter polynomial. Degree gives a cofactor of degree at most one. Thus the
Darboux identity used below captures every nearby invariant conic, not only
a chosen ansatz for its cofactor.

Seek `P F_x+Q F_y=(ell x+m y+n)F`. The ten coefficient equations, in descending monomials, are:

```
x³:    a(2−ell)+b = 0
x²y:   b(2−ell)−ma+2a(p+r)+2d = 0
x²:    2Aa+Cb−ell e−na+e+1 = 0
xy²:   d(2−ell)−mb+2ah+b(p+2r) = 0
xy:    Ab+2Cd−ell−me−nb−2a+e(p+r)+1 = 0
x:     Ae+C−ell f−ne = 0
y³:    −md+bh+2dr = 0
y²:    −m−nd−b+eh+r = 0
y:     −mf−n−e = 0
1:     −nf = 0.
```

Since `a` and `f` are nonzero near the base, the constant, linear-y and leading equations imply

`n=0`, `e=−mf`, `ell=2+t`, `b=at`,

`D=d/a=(m−2p−2r+t²)/2`,

`h=t(3m+t²−4p−6r)/4`.

The last cubic equation becomes exactly

`D(2r−m)+t h = (−m+t²+2r)(2m+t²−4p−4r)/4 = 0`.

Consequently every local invariant conic belongs to at least one of the two branches:

```
Branch I:   m=t²+2r,          D=t²−p,  h=t(t²−p).
Branch II:  m=2p+2r−t²/2,    D=t²/4,  h=t(4p−t²)/8.
```

For either choice, all remaining equations are precisely the following four equations in `(t,a,f,A)`:

```
E1 = at + mhf + m − r = 0,
E2 = 2Aa + Cat + (1+t)mf + 1 = 0,
E3 = a(At+2CD−2) + mf(m−p−r) − t − 1 = 0,
E4 = f(2+t+Am) − C = 0.
```

Their Jacobian, ordered by rows `(E1,E2,E3,E4)` and columns `(t,a,f,A)`, at the base is the SAME on both branches:

```
[ -1/2     0    0    0 ]
[ -C/2     2    0   -1 ]
[ -3/2    -2    0    0 ]
[  C/2     0    2    0 ]
```

Its determinant is **−2**. Thus each branch has exactly one nearby analytic solution `(t,a,f,A)` for every sufficiently small `(p,r)` and `C` near `C0`. Conversely, inserting either solution into the displayed reconstruction makes all ten original coefficients vanish. The cubic factorization ensures no third local branch is omitted. The result is uniform on a sufficiently small parameter neighborhood for each compact subinterval of `C>1`; no single neighborhood for the entire unbounded half-line is asserted.

For clarity about graph intersections, normalized conic coefficients are themselves locally unique for fixed full vector-field parameters. Select the eight equations with monomials `(x³,x²y,x²,xy,x,y²,y,1)` and unknowns `(a,b,d,e,f,ell,m,n)`. Their Jacobian determinant is **−2C(C+3)**, nonzero for `C>1`. The implicit-function theorem for this subsystem rules out two distinct normalized nearby conics for the same parameters.

The full projective conic matrix is nonsingular at the base: its determinant is `1/8`. Both nearby branches therefore consist of nondegenerate projective conics. Branch I is parabolic precisely on the branch intersection; its homogeneous discriminant is `a²(4p−3t²)`.

## Branch II: exact rational parabola locus

Set `k=t/2` and use `(k,m,C)` as local coordinates. Define

```
K = Cm²−6Cmk²−2Cmk+2m+12k²+16k+4,
f = 2C(1+3k)/K,
a = −(1+3k)(m+6k²+8k+2)/K,
A = [2+Cm−2Ck(3k+1)]/[2(1+3k)],
p = (−m+8k²+2k)/2,
r = m−3k²−k,
h = k(−m+6k²+2k)/2.
```

Then

`F=a(x+ky)²−fm x+y+f`, with cofactor `2(1+k)x+my`,

is exactly invariant. Direct substitution makes all ten residual coefficients identically zero. All denominators are units near `(k,m)=(0,0)` (`K=4` there).

The map `(k,m)->(p,r)` has Jacobian determinant `1/2` at the origin, and

`2p+r=k+5k²`.

Thus the local inverse is explicitly

`k=[sqrt(1+20(2p+r))−1]/10`, `m=2p+2r−2k²`.

The alternative square-root branch is not near the specified base. Independently, eliminating `a` from the reduced equations on Branch II gives

`(5t²+2t−8p−4r) [ft(t²−4p−4r)−4] = 0`.

The bracket is nonzero near the base, giving the same quadratic inverse condition and ruling out an omitted nearby parabola branch.

For compact third-order expansions let `s=2p+r`. Then

```
k = s−5s²+50s³+O(4),
h = ps−(5p+s)s²+O(4),
A = 1−Cp−3s + (C+24)s²+3Cps
    −(13C+267)s³−24Cps²+O(4).
```

Here `O(4)` is total degree at least four in `(p,r)`, locally uniformly in `C`. In particular,

`h=2p²+pr−28p³−32p²r−11pr²−r³+O(4)`.

If `alpha1` denotes the coefficient of epsilon along `p=epsilon p1`, `r=epsilon r1`, this yields exactly `alpha1=−(C+6)p1−3r1`. It does not assert that this compensation removes nonlinear displacement away from the exact invariant-conic locus.

## Branch I: exact scalar implicit equation and uniqueness

The four-equation analytic branch above is already an exact, nonsingular specification. Its `t` coordinate is equivalently the unique small solution of the following polynomial equation `B(t,p,r,C)=0`:

```
B = 2Ct⁵+2Ct⁴−3Ct³p+3Ct³r−2Ct²p+4Ct²r
    −6Ctpr−2Ctr²−4Cpr−2t³+3t²−8tr+2t−4r.
```

Indeed, `B_t(0,0,0,C)=2`, so this scalar equation has a unique small analytic root. A precise elimination justification avoiding division at exceptional points is as follows. Off algebraic denominators, solve the linear equations `E1=0` and `2E3−tE2=0` for `(a,f)`, then solve `E2=0` for `A`. The numerator of `E4` factors, up to normalization sign, as

`(3t²−4p) F1 B`, where

`F1=Ct⁴−Ct²p+Ct²r+Cr²+t²+t+r`.

On the analytic four-equation branch, `t=2r+O(2)`. Hence the first factor has initial term `−4p`, and `F1` has initial term `3r`; neither is the zero analytic germ. The denominator factors from this elimination have initial terms `2r`, `2r`, `6p`, and `−6pr` respectively, so they also are not identically zero on this germ. Therefore the numerator identity extends analytically, and the integral-domain property of real-analytic germs implies `B=0` identically on this branch. Uniqueness of the small scalar root then identifies its `t`. Values of `(a,f,A)` at exceptional denominator loci are specified by the nonsingular four-equation system, not by dividing by a vanishing expression.

Its beginning expansions are

```
t = 2r+2Cpr+2r²+O(3),
h = −2pr−2Cp²r−2pr²+8r³+O(4),
A = 1+Cp−3r−7Cpr+(6−5C)r²+O(3).
```

A useful explicit exact section is `r=0`:

```
t=m=h=e=b=0,  A=1+Cp,  f=C/2,
a=−1/[2(1+Cp)],  d=p/[2(1+Cp)],  ell=2, n=0.
```

This has invariant conic

`−x²/[2(1+Cp)] + p y²/[2(1+Cp)] + y + C/2 = 0`.

For `p!=0` it is not a parabola. In particular, conic invariance on this branch does not by itself imply a complete outer transition with the asymptotics of the base parabola; behavior at projective infinity changes.

## Exact intersection of the two graphs

By local uniqueness of the conic at fixed full parameters, the two parameter graphs intersect exactly when both cubic factors vanish. Their intersection has the following local parametrization by `(t,C)`:

```
p = 3t²/4,
r = t/2−t²/4,
h = t³/4,
A = (1−Ct²/2)/(1+3t/2),
m = t+t²/2.
```

This follows by combining `m=t²+2r`, `m=2p+2r−t²/2`, and the Branch-II inverse equation. It includes the base curve `t=0`, `C>1`.

## Reproduction

`benchmarks/hilbert16_invariant_conics.py` independently builds the original vector field and conic, checks all ten coefficients, both reduced determinants, the eight-equation determinant, the exact rational parabola formulas, the exact `r=0` Branch-I section, and the Branch-I elimination factorization using SymPy exact arithmetic. Its run returned **VERIFIED**.

The branch Taylor coefficients follow by exact coefficient matching in the four nonsingular equations. The benchmark also checks the displayed parabola expansion through total degree three. No Lean pass is claimed.


The analytic coordinates `delta_A=A-A_parabola(p,r,C)` and
`delta_h=h-h_parabola(p,r)` vanish exactly on the parabola branch.
Together with `(p,r,C)` they give a local coordinate system on the full
five-parameter field family: the derivatives in A and h form the identity
matrix. Their leading orders reproduce the compensated first direction
and the second-order logarithmic obstruction. These coordinates isolate
the invariant-parabola surface without assuming that it joins the two
particular zero-height sections of the passage problem.

Run `python benchmarks/hilbert16_invariant_conics.py`; its JSON is written
to `artifacts/hilbert16/invariant-conics.json`. Exact checks stay active under
`python -O`. The [full Hilbert program](HILBERT16.md) retains the open
return-map, all-degree and curve/surface obligations.
