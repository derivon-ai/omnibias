# The unique polynomial obstruction for the regular-parabola operator

Let K=Q(C), q=(x^2+2*x+C)/2, and let L:K[x]->K[x] be

    L(V)=q*V'-2*x*V.

The same arguments hold at every fixed real C!=0,-3, in particular every C>1.
No convergence of a formal invariant graph is claimed.

## Exact finite-dimensional calculation

For every nonnegative integer d,

    L(x^d)=((d-4)/2)*x^(d+1)+d*x^d+(C*d/2)*x^(d-1),

where the last term is zero at d=0. The only leading-degree resonance is
d=4. Consequently L preserves P4, the polynomials of degree at most four.
In the ordered basis (1,x,x^2,x^3,x^4), with input basis vectors in columns,

    M = [ 0    C/2    0     0      0  ]
        [-2     1     C     0      0  ]
        [ 0   -3/2    2    3C/2    0  ]
        [ 0     0    -1     3     2C  ]
        [ 0     0     0    -1/2    4  ],

    det(M)=8*C*(C+3).

Thus L:P4->P4 is invertible for all C>1. In particular the degree-four
leading-term cancellation does not create a polynomial kernel.

## Polynomial cokernel theorem and constructive proof

**Theorem.** Every F in K[x] has a unique representation

    F=L(V)+b(F)*x^5, with V in K[x] and b(F) in K.

Equivalently, K[x]/L(K[x]) is a one-dimensional K-vector space represented
by x^5. This is a vector-space cokernel, not a quotient ring: L(K[x]) is
not asserted to be an ideal.

**Existence.** If the residual has degree n>=6 and leading coefficient a,
subtract

    L(2*a*x^(n-1)/(n-5)).

The resulting degree is smaller than n, because the leading coefficient of
L(x^(n-1)) is (n-5)/2. Repeat until the residual has degree at most five.
Its x^5 coefficient is b. The remaining polynomial belongs to P4 and has a
unique L-primitive there by invertibility of M. Combining all subtracted
primitives proves existence. The procedure terminates after finitely many
degree reductions.

**Uniqueness.** Suppose L(W)=b*x^5. If deg(W)>=5, L(W) has degree
deg(W)+1>=6, contradiction. Otherwise W is in P4 and L(W) is in P4, so
b=0. Invertibility on P4 then gives W=0. This also proves injectivity of
L on all polynomials.

Therefore an exact polynomial solution of L(V)=F exists if and only if
b(F)=0, and is then unique. One can apply this criterion to successive
polynomial forcing terms without numerical integration.

## A scalar recurrence and an exact generating function

The map b is K-linear. Put b_n=b(x^n). Then

    b_0=...=b_4=0, b_5=1,
    b_n=-(n-1)*(2*b_(n-1)+C*b_(n-2))/(n-5), n>=6.

This follows by applying b to L(x^(n-1))=0 in the cokernel. For example,

    b_6=-10,
    b_7=60-3*C,
    b_8=112*C/3-280,
    b_9=6*C^2-808*C/3+1120.

Hence b(F) can be evaluated by a scalar recurrence before constructing V.
Equivalently, define the formal power series Theta_C(t) by

    Theta_C(0)=1,
    Theta_C'=-4*Theta_C/(1+2*t+C*t^2).

All its coefficients are polynomials in C with rational coefficients. Then

    sum[k>=0] b_(k+5)*t^k = Theta_C(t)/(1+2*t+C*t^2)^3.

Indeed the right side B satisfies

    (1+2*t+C*t^2)*B'=-(10+6*C*t)*B, B(0)=1,

whose coefficient recurrence is exactly the one above. This is an equality
of formal series; constructing any finite coefficient requires only exact
polynomial arithmetic.

## Analytic meaning of the obstruction when C>1

For real C>1 define

    D(x)=exp(4*(atan((x+1)/sqrt(C-1))-pi/2)/sqrt(C-1)),
    c=exp(-4*pi/sqrt(C-1)).

Then D'/D=2/q, D(+infinity)=1, D(-infinity)=c, and exactly

    (D*V/q^2)'=D*L(V)/q^3.

For the unique decomposition F=L(V)+b*x^5, it follows that

    integral[-R,R] D*F/q^3
      = [D*V/q^2]_-R^R + b*integral[-R,R] D*x^5/q^3.

The remaining integral has tails 8/x+O(x^-2) at positive infinity and
8*c/x+O(x^-2) at negative infinity. Therefore

    integral[-R,R] D*F/q^3 - [D*V/q^2]_-R^R
       =8*b*(1-c)*log(R)+O(1).

Thus b is precisely the remaining logarithmic coefficient divided by
8*(1-c), after subtracting the explicit boundary contribution. For high
degree F the latter can have power growth, which must not be omitted.
The bounds are uniform for C in compact subsets of (1,infinity) and bounded
coefficients of a fixed-degree family of F.

Near positive infinity, t=1/x gives D(x)=Theta_C(t). The coefficient of
x^-1 in D*F/q^3 is exactly 8*b(F), in agreement with the formal generating
function. There is no x^-1 term in the derivative of D*V/q^2: its convergent
Laurent expansion differentiates term by term near infinity, and the
constant Laurent term differentiates to zero.

## Relation to the second-order compensation calculation

The [compensation note](HILBERT16-COMPENSATION.md) defines the
degree-five second forcing

    F=F2(U)+m1*b1,
    F2(U)=(U-h)*U'+B*U, b1=-x*(x^2-C)^2/4.

Its leading coefficient is

    [x^5]F=(p*(2*p+r)-m1)/4.

Since F has degree at most five, no higher-degree reduction is needed, so

    b(F)=(p*(2*p+r)-m1)/4.

The logarithmic integral coefficient is consequently
2*(1-c)*(p*(2*p+r)-m1). Multiplication by q(R)/D(R)~R^2/2 recovers the
proved normalized second Taylor coefficient

    g2_R=(1-c)*(p*(2*p+r)-m1)*R^2*log(R)+O(R^2).

Adding an alpha2 source -alpha2*x^2 changes the polynomial primitive only:
b(-x^2)=0. A fixed alpha2 therefore cannot remove this logarithm.

## What this establishes and what it does not

The theorem supplies a complete exact primitive/obstruction decision for
each finite polynomial forcing under this operator. It isolates the one
resonant obstruction at every polynomial degree and explains the already
computed quadratic logarithm. It does not establish that all higher-order
forcing terms of a chosen nonlinear problem are polynomial, that all their
obstructions vanish, or that a formally constructed series converges. Joint
parameter/cutoff bounds, singular-map matching, return-map cyclicity, and
full Hilbert 16 remain unresolved.

The executable `benchmarks/hilbert16_polynomial_obstruction.py` checks the determinant, performs exact
reductions of x^n for n=0,...,20, reconstructs independently specified
primitives up to degree 18, verifies the second-order forcing coefficient,
and checks the first thirteen generating coefficients. It uses exact SymPy
arithmetic and no floating-point oracle.

Run `python benchmarks/hilbert16_polynomial_obstruction.py`. The JSON is written to `artifacts/hilbert16/polynomial-obstruction.json`. All checks remain active under `python -O`.
